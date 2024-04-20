from __future__ import annotations

from lightning.pytorch.callbacks import Callback
from pynput import keyboard
from textwrap import dedent
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lightning.pytorch import LightningModule, Trainer
    from pynput.keyboard import Key


class KeyboardInterruptTrialCallback(Callback):
    def __init__(self, interrupt: Key = keyboard.Key.f10, tuning: bool = False):
        """Initialize the interrupt callback.

        Args:
            key: The keyboard key to interrupt training or tuning.
            tuning: Differentiate between training or tuning.

        """

        self.interrupt = interrupt
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.interrupted = False
        self.tuning = tuning

    def on_press(self, key: Key) -> bool:
        """Stop training if the interrupt key was pressed.

        Args:
            key: The keyboard key that was pressed.

        Returns:
            True if the interrupt key was pressed, False otherwise.

        """

        if key == self.interrupt:
            message = f"""
                Interrupt key ({self.interrupt}) pressed.
                Stopping training after current epoch.
            """

            message = dedent(message)
            print(message)

            self.interrupted = True
            return False

        return True

    def on_train_start(self, _trainer: Trainer, _module: LightningModule) -> None:
        """The beginning of training.

        Args:
            _trainer: The trainer instance.
            _module: The training module instance.

        """

        self.listener.start()
        self.interrupted = False

    def on_train_epoch_end(self, trainer: Trainer, _module: LightningModule) -> None:
        """The end of each training epoch.

        Args:
            trainer: The trainer instance.
            _module: The training module instance.

        """

        if self.interrupted:
            if not self.tuning:
                self.listener.stop()

            trainer.should_stop = True
