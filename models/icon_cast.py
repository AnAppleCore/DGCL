import logging

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.base import BaseLearner
from utils.inc_net import ICONCASTNet
from utils.toolkit import tensor2numpy

num_workers = 8


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        if args.get("dataset", "").lower() != "core50":
            raise ValueError("ICON-CAST currently supports only Core50.")
        if args.get("domain_incremental", False):
            raise ValueError("ICON-CAST does not support domain_incremental mode.")
        if not args.get("enable_dgil", False):
            raise ValueError("ICON-CAST requires enable_dgil=true.")

        self._network = ICONCASTNet(args=args, pretrained=True)
        self.batch_size = int(args["batch_size"])
        self.init_lr = float(args["init_lr"])
        self.weight_decay = float(args.get("weight_decay") or 0.0)
        self.min_lr = float(args.get("min_lr") or 1e-8)
        self.num_freeze_epochs = int(args["num_freeze_epochs"])
        self.ema_decay = float(args["ema_decay"])
        self.cast_beta = float(args.get("cast_beta", 0.05))
        self.cast_clusters = int(args.get("cast_clusters", 3))
        self.cast_eps = float(args.get("cast_eps", 1e-12))
        self.cast_random_state = int(args["seed"])
        self.shift_pool = []
        self._task_start_adapters = None
        self._cluster_assignments = None
        self._cluster_centers = None
        self._cast_history = None
        self._last_cast_gradient_norm = 0.0

        if self.cast_clusters < 1:
            raise ValueError("cast_clusters must be at least 1.")
        if self.cast_beta < 0:
            raise ValueError("cast_beta must be non-negative.")

        for parameter in self._network.backbone.parameters():
            parameter.requires_grad_(False)

        logging.info(
            "ICON-CAST configuration: beta=%.6f, clusters=%d, similarity=cosine, "
            "distance_weight_grad=detached, activation=history>K, kmeans_seed=%d",
            self.cast_beta,
            self.cast_clusters,
            self.cast_random_state,
        )

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self.topk = min(self.topk, self._total_classes)
        self._network.update_fc(self._total_classes)
        self.data_manager = data_manager
        logging.info(
            "Learning on classes %d-%d from reference domain %s",
            self._known_classes,
            self._total_classes - 1,
            data_manager.get_cur_domain(self._cur_task),
        )

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes), source="train", mode="train"
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _unwrap_network(self):
        return self._network.module if isinstance(self._network, nn.DataParallel) else self._network

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        network = self._unwrap_network()
        network.attach_pets_vit(network.pets)
        self._task_start_adapters = network.snapshot_adapters().to(self._device)
        self._prepare_cast_clusters()

        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer)
        self._init_train(train_loader, test_loader, optimizer, scheduler)

        final_shift = (network.flatten_adapters() - self._task_start_adapters).detach().cpu()
        if not torch.isfinite(final_shift).all():
            raise FloatingPointError("Non-finite task adapter shift detected.")
        self.shift_pool.append(final_shift)
        logging.info(
            "ICON-CAST task %d stored shift: norm=%.8f, history_size=%d",
            self._cur_task,
            final_shift.norm().item(),
            len(self.shift_pool),
        )

    def get_optimizer(self):
        network = self._unwrap_network()
        parameters = list(network.fc.parameters()) + list(network.pets.parameters())
        optimizer_name = self.args["optimizer"].lower()
        if optimizer_name == "sgd":
            return optim.SGD(
                parameters, momentum=0.9, lr=self.init_lr, weight_decay=self.weight_decay
            )
        if optimizer_name == "adam":
            return optim.Adam(parameters, lr=self.init_lr, weight_decay=self.weight_decay)
        if optimizer_name == "adamw":
            return optim.AdamW(parameters, lr=self.init_lr, weight_decay=self.weight_decay)
        raise ValueError("Unsupported optimizer: {}".format(self.args["optimizer"]))

    def get_scheduler(self, optimizer):
        scheduler_name = self.args["scheduler"].lower()
        if scheduler_name == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=self.args["tuned_epoch"],
                eta_min=self.min_lr,
            )
        if scheduler_name == "steplr":
            return optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer,
                milestones=self.args["init_milestones"],
                gamma=self.args["init_lr_decay"],
            )
        if scheduler_name == "constant":
            return None
        raise ValueError("Unsupported scheduler: {}".format(self.args["scheduler"]))

    def _prepare_cast_clusters(self):
        self._cluster_assignments = None
        self._cluster_centers = None
        self._cast_history = None
        history_size = len(self.shift_pool)
        if self.cast_beta == 0 or history_size <= self.cast_clusters:
            logging.info(
                "ICON-CAST task %d inactive: history_size=%d, requires history_size>%d, beta=%.6f",
                self._cur_task,
                history_size,
                self.cast_clusters,
                self.cast_beta,
            )
            return

        history = torch.stack(self.shift_pool).float()
        kmeans = KMeans(
            n_clusters=self.cast_clusters,
            n_init=10,
            random_state=self.cast_random_state,
        )
        assignments = kmeans.fit_predict(history.numpy())
        self._cluster_assignments = torch.as_tensor(assignments, dtype=torch.long)
        self._cluster_centers = torch.as_tensor(kmeans.cluster_centers_, dtype=torch.float32)
        self._cast_history = history
        logging.info(
            "ICON-CAST task %d active: history_size=%d, assignments=%s",
            self._cur_task,
            history_size,
            assignments.tolist(),
        )

    def _cast_loss(self):
        network = self._unwrap_network()
        current_adapters = network.flatten_adapters()
        zero = current_adapters.sum() * 0.0
        current_shift = current_adapters - self._task_start_adapters
        shift_norm = current_shift.norm()
        if self._cast_history is None:
            return zero, shift_norm.detach()

        if not torch.isfinite(shift_norm):
            raise FloatingPointError("Non-finite current adapter shift norm detected.")
        if shift_norm.detach().item() <= self.cast_eps:
            return zero, shift_norm.detach()

        centers = self._cluster_centers.to(current_shift.device, current_shift.dtype)
        current_cluster = torch.argmin(torch.norm(centers - current_shift.detach(), dim=1))
        other_mask = self._cluster_assignments != current_cluster.cpu()
        if not torch.any(other_mask):
            return zero, shift_norm.detach()

        other_shifts = self._cast_history[other_mask].to(current_shift.device, current_shift.dtype)
        distances = torch.norm(other_shifts - current_shift.detach().unsqueeze(0), dim=1)
        distance_sum = distances.sum()
        if distance_sum.item() <= self.cast_eps:
            weights = torch.full_like(distances, 1.0 / len(distances))
        else:
            weights = distances / distance_sum
        weights = weights.detach()

        cosine = F.cosine_similarity(
            current_shift.unsqueeze(0), other_shifts, dim=1, eps=self.cast_eps
        )
        cast_raw = torch.abs(torch.sum(weights * cosine))
        cast_loss = self.cast_beta * cast_raw
        if not torch.isfinite(cast_loss):
            raise FloatingPointError("Non-finite CAST loss detected.")
        return cast_loss, shift_norm.detach()

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        progress = tqdm(range(self.args["tuned_epoch"]))
        network = self._unwrap_network()
        network.attach_pets_vit(network.pets)
        for parameter in network.pets.parameters():
            parameter.requires_grad_(False)

        for epoch in progress:
            self._network.train()
            network.fc.train()
            if epoch == self.num_freeze_epochs:
                for parameter in network.pets.parameters():
                    parameter.requires_grad_(True)
                logging.info("ICON-CAST task %d unfroze adapters at epoch %d", self._cur_task, epoch + 1)

            ce_sum = 0.0
            cast_sum = 0.0
            total_loss_sum = 0.0
            shift_norm_sum = 0.0
            correct, total = 0, 0
            for _, inputs, targets in train_loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                output = self._network(inputs)
                logits = output["logits"][:, :self._total_classes]
                logits = logits.clone()
                logits[:, :self._known_classes] = float("-inf")

                ce_loss = F.cross_entropy(logits, targets.long())
                cast_loss, shift_norm = self._cast_loss()
                loss = ce_loss + cast_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite ICON-CAST total loss detected.")

                optimizer.zero_grad()
                self._last_cast_gradient_norm = 0.0
                if cast_loss.detach().item() > 0:
                    cast_gradients = torch.autograd.grad(
                        cast_loss,
                        tuple(network.pets.parameters()),
                        retain_graph=True,
                        allow_unused=True,
                    )
                    self._last_cast_gradient_norm = sum(
                        gradient.detach().pow(2).sum().item()
                        for gradient in cast_gradients
                        if gradient is not None
                    ) ** 0.5
                loss.backward()
                optimizer.step()

                ce_sum += ce_loss.item()
                cast_sum += cast_loss.detach().item()
                total_loss_sum += loss.item()
                shift_norm_sum += shift_norm.item()
                predictions = torch.argmax(logits, dim=1)
                correct += predictions.eq(targets).cpu().sum()
                total += len(targets)

                with torch.no_grad():
                    for parameter, parameter_ema in zip(
                        network.pets.parameters(), network.pets_emas.parameters()
                    ):
                        parameter_ema.mul_(self.ema_decay).add_(
                            parameter, alpha=1.0 - self.ema_decay
                        )

            if scheduler:
                scheduler.step()

            batches = max(1, len(train_loader))
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            metrics = (
                ce_sum / batches,
                cast_sum / batches,
                total_loss_sum / batches,
                shift_norm_sum / batches,
            )
            if epoch + 1 == self.args["tuned_epoch"]:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = (
                    "Task {}, Epoch {}/{} => CE {:.4f}, CAST {:.6f}, Loss {:.4f}, "
                    "Shift {:.6f}, CAST_grad {:.6f}, Train_accy {:.2f}, Test_accy {:.2f}"
                ).format(
                    self._cur_task,
                    epoch + 1,
                    self.args["tuned_epoch"],
                    *metrics,
                    self._last_cast_gradient_norm,
                    train_acc,
                    test_acc,
                )
            else:
                info = (
                    "Task {}, Epoch {}/{} => CE {:.4f}, CAST {:.6f}, Loss {:.4f}, "
                    "Shift {:.6f}, CAST_grad {:.6f}, Train_accy {:.2f}"
                ).format(
                    self._cur_task,
                    epoch + 1,
                    self.args["tuned_epoch"],
                    *metrics,
                    self._last_cast_gradient_norm,
                    train_acc,
                )
            progress.set_description(info)
            logging.info(info)

    def _ensemble_probabilities(self, model, inputs):
        network = model.module if isinstance(model, nn.DataParallel) else model
        network.attach_pets_vit(network.pets)
        with torch.no_grad():
            online = model(inputs)["logits"][:, :self._total_classes].softmax(dim=1)
        try:
            network.attach_pets_vit(network.pets_emas)
            with torch.no_grad():
                ema = model(inputs)["logits"][:, :self._total_classes].softmax(dim=1)
        finally:
            network.attach_pets_vit(network.pets)
        return torch.stack([online, ema], dim=-1).max(dim=-1)[0]

    def _eval_cnn(self, loader):
        self._network.eval()
        predictions, targets_all = [], []
        for _, inputs, targets in loader:
            inputs = inputs.to(self._device)
            outputs = self._ensemble_probabilities(self._network, inputs)
            topk = torch.topk(
                outputs, k=min(self.topk, outputs.shape[1]), dim=1, largest=True, sorted=True
            )[1]
            predictions.append(topk.cpu().numpy())
            targets_all.append(targets.cpu().numpy())
        return np.concatenate(predictions), np.concatenate(targets_all)

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for _, inputs, targets in loader:
            inputs = inputs.to(self._device)
            outputs = self._ensemble_probabilities(model, inputs)
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions.cpu() == targets).sum()
            total += len(targets)
        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)
