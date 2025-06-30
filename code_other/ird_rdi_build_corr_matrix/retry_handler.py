import re
import sys
import time
import traceback
from datetime import datetime, timedelta


class RetryHandler:
    def __init__(self, method, timeout=600.0, min_try=10, sleep_time=1.0, sleep_time_gain=2.0, sleep_time_max=60.0, verbose=True, handled_errors=(OSError,), raise_errors=(FileNotFoundError,)):
        """Launch a method with retry handler.

        :param builtins.function method: The method to launch.
        :param float timeout: The global fail timeout, in seconds.
        :param int min_try: Minimum number of tries, regardless of the timeout.
        :param float sleep_time: The base sleep time between retries, in seconds.
        :param float sleep_time_gain: Multiply the sleep time by this factor each try.
        :param float sleep_time_max: The maximum sleep time between tries, in seconds.
        :param bool verbose: Display a log line on success if at least a try.
        :param Exception|Error|tuple[Exception|Error] handled_errors: The handled errors. ``(default: OSError)``
        :param Exception|Error|tuple[Exception|Error] raise_errors: Always raise those errors. (ie. if handled separately) ``(default: FileNotFoundError)``
        """
        self.method = method
        self.timeout = timeout
        self.min_try = min_try
        self.sleep_time = sleep_time
        self.slee_time_gain = sleep_time_gain
        self.max_sleep_time = sleep_time_max
        self.verbose = verbose
        if type(handled_errors) == tuple:
            self.handled_errors = handled_errors  # type: tuple[Exception]
        elif handled_errors is not None:
            self.handled_errors = (handled_errors,)  # type: tuple[Exception]
        elif handled_errors is None:
            self.handled_errors = ()  # type: tuple[Exception]

        if type(raise_errors) == tuple:
            self.raise_errors = raise_errors  # type: tuple[Exception]
        elif raise_errors is not None:
            self.raise_errors = (raise_errors,)  # type: tuple[Exception]
        elif raise_errors is None:
            self.raise_errors = ()  # type: tuple[Exception]

        self.start_time = datetime.now()
        self.timeout_time = self.start_time + timedelta(seconds=self.timeout)
        self.try_nb = 0
        self.ok = False
        self.last_error = None
        self.log_time_format = '%Y-%m-%d %H:%M:%S'
        self.args = []
        self.kwargs = {}

    def run(self, *args, **kwargs):
        """Run the method with the given arguments.

        :return: The method results.
        :rtype: Any
        """
        self.apply(*args, **kwargs)
        return self.launch()

    def apply(self, *args, **kwargs):
        """Apply these arguments to the method.

        :return: The handler object.
        :rtype: RetryHandler
        """
        self.args = args
        self.kwargs = kwargs
        return self

    def method_label(self, display_args=True):
        """Return the method label.

        :param bool display_args: Display the method arguments.
        :return: The label.
        :rtype: str
        """
        if getattr(self.method, "__name__", None) is not None:
            label = self.method.__name__
        else:
            label = str(self.method)

        if display_args:
            args_labels = list(map(lambda x: str(x), self.args))
            for k, v in self.kwargs.items():
                args_labels.append("%s=%s" % (k, str(v)))

            label = "%s(%s)" % (label, ", ".join(args_labels))

        return label

    def launch(self):
        """Run the method.

        :return: The method results.
        :rtype: Any
        """

        ret = None
        while not self.ok:
            # Rety loop
            self.try_nb = self.try_nb + 1

            try:
                ret = self.method(*self.args, **self.kwargs)
                self.ok = True

            except Exception as e:
                handled = False

                # retry on those
                for handled_error in self.handled_errors:
                    if isinstance(e, handled_error):
                        handled = True
                        break

                # alway raise on those
                for raise_error in self.raise_errors:
                    if isinstance(e, raise_error):
                        handled = False
                        break

                if not handled:
                    raise e

                if self.verbose:
                    print("Retry handler [%s]: failed call %s of %s total %s wait. Error: %s." % (
                        datetime.now().strftime(self.log_time_format),
                        self.try_nb,
                        self.method_label(),
                        timedelta_label(datetime.now() - self.start_time),
                        repr(e)
                    ), file=sys.stderr, flush=True)

                self.last_error = e

                # check global timeout and minimum try
                if datetime.now() > self.timeout_time and self.try_nb >= self.min_try:
                    break

                # wait
                time.sleep(self.sleep_time)

                # double sleep time, until max
                self.sleep_time = min(self.sleep_time * self.slee_time_gain, self.max_sleep_time)

        if not self.ok and datetime.now() > self.timeout_time:
            # Timeout
            traceback.print_exception(type(self.last_error), self.last_error, self.last_error.__traceback__)

            raise TimeoutError("Retry handler [%s]: %s timeout after %s try and %s. Last error: %s." % (
                datetime.now().strftime(self.log_time_format),
                self.method_label(),
                self.try_nb,
                timedelta_label(datetime.now() - self.start_time),
                repr(self.last_error)
            ))

        elif self.ok and self.verbose and self.try_nb > 1:
            # Ok after at least one retry
            print("Retry handler [%s]: sucessful call of %s after %s try and %s wait. Last error: %s." % (
                datetime.now().strftime(self.log_time_format),
                self.method_label(),
                self.try_nb,
                timedelta_label(datetime.now() - self.start_time),
                repr(self.last_error)
            ), file=sys.stderr, flush=True)

        return ret


def timedelta_shortlabel(delta):
    """Transforms a timedelta object into a readable label: days and time number, without seconds or microseconds.

    :param timedelta delta: The timedelta.
    :rtype: str
    :return: The label.
    """
    if delta is None:
        return delta

    return re.sub("^(.*days?|\d+:\d+)?(, \d+:\d+)?.*$", "\g<1>\g<2>", str(delta))


def timedelta_label(td, show_millis=False, seconds_only=True):
    """Transform the given timedelta into a readable label.

    :param timedelta td: The timedelta.
    :param bool show_millis: Show milliseconds.
    :param bool seconds_only: Allow the display of seconds only.
    :return: The label.
    :rtype: str
    """
    try:
        minus, days, hours, minutes, seconds, millis = re.sub("^(-?)((\d+) days?, )?(\d+):(\d+):(\d+)\.?(\d{2})?.*$", "\g<1>:\g<3>:\g<4>:\g<5>:\g<6>:\g<7>", str(td)).split(":")
    except ValueError:
        return str(td)

    minus = minus == "-"
    days = int(days) if days else 0
    hours = int(hours)
    minutes = int(minutes)
    seconds = int(seconds)
    millis = int(millis) if millis else 0

    if days:
        label = "%dd %sh%02dm" % (days, hours, minutes)

    elif hours:
        label = "%sh%02dm" % (hours, minutes)

    elif show_millis:
        if minutes or not seconds_only:
            label = "%dm%02ds%02d" % (minutes, seconds, millis)
        else:
            label = "%ds%02d" % (seconds, millis)

    else:
        if minutes or not seconds_only:
            label = "%sm%02ds" % (minutes, seconds)
        else:
            label = "%ds" % (seconds,)

    if minus:
        label = "-%s" % label

    return label
