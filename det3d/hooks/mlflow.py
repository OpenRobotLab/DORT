from mmcv.runner.hooks.logger import MlflowLoggerHook
from mmcv.runner.hooks import HOOKS
from mmcv.runner.dist_utils import master_only

@HOOKS.register_module()
class CustomMlflowLoggerHook(MlflowLoggerHook):
    def __init__(self, run_name, **kwargs):
        super().__init__(**kwargs)
        # self.mlflow.log_params(cfg)
        self.run_name = run_name

    @master_only
    def before_run(self, runner):
        super().before_run(runner)
        self.mlflow.start_run(run_name=self.run_name)

    def after_run(self, runner):
        super().after_run(runner)
        self.mlflow.end_run()