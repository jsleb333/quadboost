import sys, os
sys.path.append(os.getcwd())

import warnings
import pickle as pkl
import csv
import logging

from callbacks import Callback


class SaveCallback(Callback):
    """
    This abstract callback provides the basics for a Callback which saves something. It handles the filename and directory formating and the necessary implementation for atomic writing (no information loss can occur if the process is killed during saving).
    """
    def __init__(self, filename, dirname='.', atomic_write=True, manager=None):
        """
        Args:
            filename (str): The name of the file to be saved. This string can be formated using the 'format_filename' method if it is overloaded in inherited classes.
            dirname (str, optional): The name of the directory. By default, it saves in the save directory as the script.
            atomic_write (Boolean, optional): If True, will try to save atomically the file.
            manager: See Callback documentation.
        """
        super().__init__(manager)
        self.filename = filename
        self.dirname = dirname
        os.makedirs(dirname, exist_ok=True)

        self.atomic_write = atomic_write

    @property
    def filedir(self):
        return os.path.join(self.dirname, self.format_filename(self.filename))

    @property
    def tmp_filedir(self):
        return os.path.join(self.dirname, 'tmp_' + self.format_filename(self.filename))

    def format_filename(self, filename):
        return filename

    def save(self, *args, **kwargs):
        logging.debug(f'Saving to file {self.filedir}')
        atomic_save_successful = False
        if self.atomic_write:
            atomic_save_successful = self._atomic_save(*args, **kwargs)

        if not atomic_save_successful:
            self._save(self.filedir, *args, **kwargs)

    def _atomic_save(self, *args, **kwargs):
        self._save(self.tmp_filedir, *args, **kwargs)
        try:
            os.replace(self.tmp_filedir, self.filedir)
            atomic_save_successful = True
        except OSError:
            warning_message = f"Could not replace '{self.filedir}' with '{self.tmp_filedir}'. Saving non-atomically hereafter."
            warnings.warn(warning_message)
            logging.warning(warning_message)

            os.remove(self.tmp_filedir)
            atomic_save_successful = False
            self.atomic_write = False

        return atomic_save_successful

    def _save(self, filedir, *args, **kwargs):
        raise NotImplementedError


class PeriodicSaveCallback(SaveCallback):
    """
    This class offers the possibility to save periodically by keeping a counter to the number of calls made to 'save' and saving only on the specified 'period'.
    """
    def __init__(self, *args, period=1, **kwargs):
        """
        Args:
            period (int, optional): A save will be made every 'period' times the 'save' method is called.
        """
        super().__init__(*args, **kwargs)
        self.n_calls = 0
        self.period = period

    def save(self, *args, override_period=False, **kwargs):
        """
        Periodically saves a file. If 'override_period' is True, the period is not considered
        """
        self.n_calls += 1
        saved = False
        if self.n_calls % self.period == 0 or override_period:
            super().save(*args, **kwargs)
            saved = True
        return saved


class PickleSave(SaveCallback):
    """
    Implements a saving protocol in Pickle
    """
    def __init__(self, *args, protocol=pkl.HIGHEST_PROTOCOL, open_mode='wb', **kwargs):
        super().__init__(*args, **kwargs)
        self.protocol = protocol
        self.open_mode = open_mode

    def _save(self, filedir, obj):
        with open(filedir, self.open_mode) as file:
            pkl.dump(obj, file, protocol=self.protocol)


class CSVSave(SaveCallback):
    """
    Implements a saving protocol in CSV
    """
    def __init__(self, *args, open_mode='w', delimiter=',', newline='', **kwargs):
        super().__init__(*args, **kwargs)
        self.delimiter = delimiter
        self.newline = newline
        self.open_mode = open_mode

    def _save(self, filedir, rows):
        with open(filedir, self.open_mode, newline=self.newline) as file:
            writer = csv.writer(file, delimiter=self.delimiter)
            writer.writerows(rows)
