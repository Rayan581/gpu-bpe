"""
Control layer: scheduler, staleness tracking, micro-batch assignment.

Implements staleness budgets and adaptive batch assignment for DC-ASGD.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import time
from collections import defaultdict


@dataclass
class WorkerStatus:
    """Status of a worker for staleness tracking."""
    worker_id: int
    last_update_step: int
    last_fetch_step: int
    timestamp: float
    processed_batches: int = 0


class StalnessTracker:
    """
    Tracks gradient staleness for each worker.

    Implements staleness bound enforcement for DC-ASGD:
    staleness_ij = current_step - last_update_step_i
    """

    def __init__(self, max_staleness: int = 10):
        self.max_staleness = max_staleness
        self.worker_status = {}  # worker_id -> WorkerStatus
        self.global_step = 0

    def register_worker(self, worker_id: int):
        """Register a new worker."""
        self.worker_status[worker_id] = WorkerStatus(
            worker_id=worker_id,
            last_update_step=0,
            last_fetch_step=0,
            timestamp=time.time()
        )

    def get_staleness(self, worker_id: int) -> int:
        """Get staleness of worker (steps since last gradient update)."""
        if worker_id not in self.worker_status:
            return 0
        return self.global_step - self.worker_status[worker_id].last_update_step

    def is_within_budget(self, worker_id: int) -> bool:
        """Check if worker staleness is within budget."""
        return self.get_staleness(worker_id) <= self.max_staleness

    def update_worker_step(self, worker_id: int, step: int):
        """Update worker's last update step."""
        if worker_id in self.worker_status:
            self.worker_status[worker_id].last_update_step = step
            self.worker_status[worker_id].timestamp = time.time()

    def get_worker_stats(self) -> Dict:
        """Get statistics for all workers."""
        stats = {}
        for worker_id, status in self.worker_status.items():
            stats[worker_id] = {
                'staleness': self.get_staleness(worker_id),
                'processed_batches': status.processed_batches,
                'in_budget': self.is_within_budget(worker_id)
            }
        return stats

    def increment_global_step(self):
        """Move to next global step."""
        self.global_step += 1


class MicroBatchScheduler:
    """
    Assigns micro-batches to workers based on staleness and capacity.

    Implements fair queue scheduling while respecting staleness bounds
    for DC-ASGD compensation.
    """

    def __init__(self, num_workers: int, staleness_tracker: StalnessTracker):
        self.num_workers = num_workers
        self.staleness_tracker = staleness_tracker
        self.worker_queues = defaultdict(list)  # worker_id -> [batch_ids]
        self.pending_batches = []  # unassigned batch ids
        self.batch_assignments = {}  # batch_id -> worker_id
        self.batch_counter = 0

    def add_batch(self, batch_id: Optional[int] = None) -> int:
        """Add a batch to pending queue."""
        if batch_id is None:
            batch_id = self.batch_counter
            self.batch_counter += 1

        self.pending_batches.append(batch_id)
        return batch_id

    def assign_batches(self) -> Dict[int, int]:
        """
        Assign pending batches to workers.

        Returns:
            {batch_id: worker_id}
        """
        assignments = {}

        # Assign to workers in round-robin, respecting staleness budgets
        while self.pending_batches:
            batch_id = self.pending_batches.pop(0)

            # Find best worker (least staleness, smallest queue)
            best_worker = -1
            best_score = float('inf')

            for worker_id in range(self.num_workers):
                if not self.staleness_tracker.is_within_budget(worker_id):
                    continue

                staleness = self.staleness_tracker.get_staleness(worker_id)
                queue_size = len(self.worker_queues[worker_id])
                score = staleness + 0.1 * queue_size  # Prioritize low staleness

                if score < best_score:
                    best_score = score
                    best_worker = worker_id

            if best_worker >= 0:
                self.worker_queues[best_worker].append(batch_id)
                self.batch_assignments[batch_id] = best_worker
                assignments[batch_id] = best_worker
            else:
                # No eligible worker (all over budget), re-queue
                self.pending_batches.append(batch_id)
                break

        return assignments

    def get_next_batch(self, worker_id: int) -> Optional[int]:
        """Get next batch for worker."""
        if self.worker_queues[worker_id]:
            return self.worker_queues[worker_id].pop(0)
        return None

    def get_stats(self) -> Dict:
        """Get scheduler statistics."""
        return {
            'pending_batches': len(self.pending_batches),
            'worker_queues': {
                wid: len(batches)
                for wid, batches in self.worker_queues.items()
            },
            'batch_counter': self.batch_counter
        }


class ControlLayer:
    """
    Central control layer for three-tier distributed training.

    Coordinates:
    - staleness tracking via StalnessTracker
    - micro-batch scheduling via MicroBatchScheduler
    - global parameter version tracking
    - gradient aggregation sync points
    """

    def __init__(
        self,
        num_workers: int,
        max_staleness: int = 10,
        sync_interval: int = 100
    ):
        self.num_workers = num_workers
        self.max_staleness = max_staleness
        self.sync_interval = sync_interval

        self.staleness_tracker = StalnessTracker(max_staleness=max_staleness)
        self.scheduler = MicroBatchScheduler(num_workers, self.staleness_tracker)

        # Global state
        self.global_step = 0
        self.parameter_version = 0
        self.pending_gradients = {}  # step -> {worker_id -> grad_batch}
        self.synced_steps = []

        for worker_id in range(num_workers):
            self.staleness_tracker.register_worker(worker_id)

    def submit_gradient(self, worker_id: int, step: int, grad_batch):
        """Worker submits computed gradients."""
        if step not in self.pending_gradients:
            self.pending_gradients[step] = {}

        self.pending_gradients[step][worker_id] = grad_batch
        self.staleness_tracker.update_worker_step(worker_id, step)

    def need_sync(self) -> bool:
        """Check if synchronization is needed."""
        return self.global_step % self.sync_interval == 0

    def synchronize(self) -> Dict:
        """
        Perform synchronization: aggregate pending gradients.

        Returns aggregation metadata.
        """
        self.global_step += 1
        self.staleness_tracker.increment_global_step()
        self.parameter_version += 1

        self.synced_steps.append(self.global_step)

        return {
            'global_step': self.global_step,
            'parameter_version': self.parameter_version,
            'num_pending': len(self.pending_gradients)
        }

    def get_control_state(self) -> Dict:
        """Snapshot of control layer state."""
        return {
            'global_step': self.global_step,
            'parameter_version': self.parameter_version,
            'worker_stats': self.staleness_tracker.get_worker_stats(),
            'scheduler_stats': self.scheduler.get_stats(),
            'synced_steps': len(self.synced_steps)
        }


def test_control_layer():
    """Test control layer functionality."""
    control = ControlLayer(num_workers=2, max_staleness=5, sync_interval=3)

    # Simulate batch assignments
    for i in range(10):
        batch_id = control.scheduler.add_batch()
        print(f"Added batch {batch_id}")

    assignments = control.scheduler.assign_batches()
    print(f"Assignments: {assignments}\n")

    # Simulate worker gradient submissions
    for step in range(5):
        print(f"Step {step}:")
        if step >= 1:
            control.submit_gradient(0, step - 1, f"grad_batch_0_{step-1}")
        if step >= 2:
            control.submit_gradient(1, step - 1, f"grad_batch_1_{step-1}")

        print(f"  Staleness: {control.staleness_tracker.get_worker_stats()}")

        if control.need_sync():
            print(f"  SYNC!")
            control.synchronize()

        print()


if __name__ == "__main__":
    test_control_layer()
