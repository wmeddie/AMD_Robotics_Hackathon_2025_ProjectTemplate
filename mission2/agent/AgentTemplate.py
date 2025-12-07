from argparse import ArgumentParser
from xai_components.base import SubGraphExecutor, InArg, OutArg, Component, xai_component, parse_bool
from xai_components.xai_agent.agent_components import AgentDefineTool, AgentNumpyMemory, AgentRun, AgentInit, AgentToolOutput, AgentMakeToolbelt
from xai_components.xai_converse.converse_components import ConverseDefineAgent, ConverseEmitResponse, ConverseRun, ConverseMakeServer
from xai_components.xai_lerobot.lerobot_components import RunPolicyComponent
from xai_components.xai_openai.openai_components import OpenAIAuthorize
from xai_components.xai_utils.utils import Print

@xai_component(type='xircuits_workflow')
class AgentTemplate(Component):

    def __init__(self, id: str=None):
        super().__init__()
        self.__id__ = id
        self.__start_nodes__ = []
        self.c_0 = AgentDefineTool()
        self.c_0.__id__ = '972bf2ed-5217-4b6d-9051-860a6c1e847e'
        self.c_1 = ConverseDefineAgent()
        self.c_1.__id__ = '72f82eb9-cd83-411b-8c67-37b1acd73b63'
        self.c_2 = AgentRun()
        self.c_2.__id__ = 'd24f8b95-46d9-4dc7-9390-3f7e0c2a687c'
        self.c_3 = ConverseEmitResponse()
        self.c_3.__id__ = 'e43c7caa-26de-4543-9bfc-0c15f5449612'
        self.c_4 = Print()
        self.c_4.__id__ = 'b4361e28-d561-4d7a-bcf7-2bae79ec332f'
        self.c_5 = AgentInit()
        self.c_5.__id__ = 'c5053115-d90a-444c-9777-72f9ef6eadf2'
        self.c_6 = ConverseRun()
        self.c_6.__id__ = 'd23d0fe4-5b24-423c-9e86-b7f0524bf700'
        self.c_7 = AgentNumpyMemory()
        self.c_7.__id__ = '814a8084-69f0-4cab-bcaa-b2b60dd8419c'
        self.c_8 = RunPolicyComponent()
        self.c_8.__id__ = '9576ee8e-b49a-49a8-9f08-dcfee1e91518'
        self.c_9 = AgentToolOutput()
        self.c_9.__id__ = 'b3bbdf9a-bdfe-453f-91f8-80bfeb039bd5'
        self.c_10 = OpenAIAuthorize()
        self.c_10.__id__ = 'a201a404-bd23-487f-8f92-422b9de01cdb'
        self.c_11 = AgentMakeToolbelt()
        self.c_11.__id__ = '363608ce-4ae1-40e7-a24b-bebd31c480e5'
        self.c_12 = AgentToolOutput()
        self.c_12.__id__ = '6969c10e-ae9d-4dd8-9ae7-cd85046cd33c'
        self.c_13 = RunPolicyComponent()
        self.c_13.__id__ = '448c613c-6504-4090-8308-19415b49a4bf'
        self.c_14 = AgentDefineTool()
        self.c_14.__id__ = '8571d4b0-d9be-43da-a3e2-5d47777f6474'
        self.c_15 = ConverseMakeServer()
        self.c_15.__id__ = '9c595812-f7f4-41fa-a623-79625fa890f4'
        self.c_0.tool_name.value = 'place_rock'
        self.c_0.description.value = 'Places the rock into the zen garden completing it.'
        self.c_1.name.value = 'default'
        self.c_2.agent_name.value = 'default_agent'
        self.c_2.conversation.connect(self.c_1.conversation)
        self.c_3.value.connect(self.c_2.last_response)
        self.c_4.msg.connect(self.c_2.last_response)
        self.c_5.agent_name.value = 'default_agent'
        self.c_5.agent_provider.value = 'openai'
        self.c_5.agent_model.value = 'gpt-oss-120b'
        self.c_5.agent_memory.connect(self.c_7.memory)
        self.c_5.system_prompt.value = "You are a robot assistant called Xaibo. You have access to tools that help guide the robot to accomplish the user's goals.  Based on the message from the user break down the task step-by-step and use the tools to have the robot execute that step.\n\n{tool_instruction}\n\nYou have access to the following tools: \n\n{memory}\n{tools}\n"
        self.c_5.max_thoughts.value = 10
        self.c_5.toolbelt_spec.connect(self.c_11.toolbelt_spec)
        self.c_8.checkpoint.value = 'wmeddie/smolvla_place_rock3_from_base'
        self.c_8.duration.value = 300
        self.c_8.hz.value = 10
        self.c_8.task.value = 'Place Rock'
        self.c_9.results.value = ['Complete']
        self.c_10.base_url.value = 'https://relay.public.cloud.xpress.ai'
        self.c_10.api_key.value = 'opensesame'
        self.c_10.from_env.value = False
        self.c_12.results.value = ['Complete']
        self.c_13.checkpoint.value = 'wmeddie/smolvla_rake1_from_base'
        self.c_13.duration.value = 300
        self.c_13.hz.value = 10
        self.c_13.task.value = 'Rake Sand'
        self.c_14.tool_name.value = 'rake_sand'
        self.c_14.description.value = 'Rakes the sand to provide the backdrop for foreground objects like rocks or stamps'
        self.c_15.auth_token.value = 'opensesame'
        self.c_15.max_timeout.value = 90
        self.c_0.next = self.c_8
        self.c_1.next = self.c_2
        self.c_2.next = self.c_3
        self.c_2.on_thought = SubGraphExecutor(self.c_4)
        self.c_3.next = None
        self.c_4.next = None
        self.c_5.next = self.c_6
        self.c_6.next = None
        self.c_7.next = self.c_5
        self.c_8.next = self.c_9
        self.c_9.next = None
        self.c_10.next = self.c_15
        self.c_11.next = self.c_7
        self.c_12.next = None
        self.c_13.next = self.c_12
        self.c_14.next = self.c_13
        self.c_15.next = self.c_11
        self.__start_nodes__.append(self.c_0)
        self.__start_nodes__.append(self.c_1)
        self.__start_nodes__.append(self.c_14)

    def execute(self, ctx):
        for node in self.__start_nodes__:
            if hasattr(node, 'init'):
                node.init(ctx)
        SubGraphExecutor(self.c_10).do(ctx)

def main(args):
    import pprint
    ctx = {}
    ctx['args'] = args
    flow = AgentTemplate()
    flow.next = None
    flow.do(ctx)
if __name__ == '__main__':
    parser = ArgumentParser()
    args, _ = parser.parse_known_args()
    main(args)
    print('\nFinished Executing')