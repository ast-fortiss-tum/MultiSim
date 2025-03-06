class RestrictiveConfigException(Exception):
    def __init__(self, message="Search configuration is too restrictive, no time for search available."):
        self.message = message
        super().__init__(self.message)
