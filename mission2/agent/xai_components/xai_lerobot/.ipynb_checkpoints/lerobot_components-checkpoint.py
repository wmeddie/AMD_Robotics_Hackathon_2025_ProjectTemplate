from xai_components.base import InArg, OutArg, InCompArg, Component, BaseComponent, xai_component, dynalist, SubGraphExecutor
import os
# Runs
# python test_policy.py --checkpoint wmeddie/smolvla_place_rock3_from_base --duration 600 --hz 30 --task "Place Rock"

@xai_component
class RunPolicyComponent(Component):
    checkpoint: InArg[str]
    duration: InArg[int]
    hz: InArg[int]
    task: InArg[str]

    def execute(self, ctx) -> None:
        checkpoint = self.checkpoint.value
        duration = self.duration.value
        hz = self.hz.value
        task = self.task.value

        os.system(f'python test_policy.py --checkpoint {checkpoint} --duration {duration} --hz {hz} --task "{task}"')

