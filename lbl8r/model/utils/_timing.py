import contextlib, time


class Timing(contextlib.ContextDecorator):
    def __init__(self, prefix="model", enabled=True):
        self.prefix = prefix
        self.enabled = enabled
        self.dynamic_prefix = None

    def __enter__(self):
        self.st = time.time()

    def __exit__(self, *exc):
        self.et = time.time() - self.st
        prefix = self.dynamic_prefix if self.dynamic_prefix else self.prefix
        if self.enabled:
            print(f"           🧭🧭🧭 {prefix} {self._format_time(self.et)}🧭🧭🧭 ")

    def __call__(self, func):
        def wrapped_func(*args, **kwargs):
            # Use a dynamic prefix if specified
            if self.prefix is not None and self.prefix in kwargs:
                self.dynamic_prefix = kwargs[self.prefix]
            else:
                self.dynamic_prefix = None
            with self:
                return func(*args, **kwargs)

        return wrapped_func

    def _format_time(self, seconds):
        if seconds < 60:
            return f"training time: {seconds:.2f} sec "
        elif seconds < 3600:
            minutes, seconds = divmod(seconds, 60)
            return f"training time: {int(minutes)}:{seconds:.2f} (m:s)"
        else:
            hours, remainder = divmod(seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"training time: {int(hours)}:{int(minutes)}:{seconds:.2f} (h:m:s)"
