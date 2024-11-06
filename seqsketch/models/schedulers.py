from diffusers.schedulers import DDIMScheduler


class myDDIMScheduler(DDIMScheduler):
    def __init__(self, params):
        super().__init__(**params.__dict__)
