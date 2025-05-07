
import minigrid
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal
from minigrid.minigrid_env import MiniGridEnv
import gymnasium as gym

class EmptyEnvRandom10x10(MiniGridEnv):
    """
    Empty 10x10 grid environment with a random goal location and fixed agent start.
    """
    def __init__(self, size=10, **kwargs):
        # Agent always starts at (1, 1) facing right (direction 0)
        self.agent_start_pos = (1, 1)
        self.agent_start_dir = 0
        mission_space = MissionSpace(mission_func=lambda: "get to the green goal square")
        super().__init__(
            mission_space=mission_space,
            width=size,
            height=size,
            max_steps=4 * size * size,
            **kwargs,
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent at the fixed starting position
        self.agent_pos = self.agent_start_pos
        self.agent_dir = self.agent_start_dir

        # Place a goal square randomly within the grid, excluding walls and agent start
        self.place_obj(Goal(), top=(1,1), size=(width-2, height-2))

        self.mission = "get to the green goal square"

# Register the custom environment
gym.register(
    id="MiniGrid-Empty-Random-10x10-v0",
    entry_point="train_minigrid:EmptyEnvRandom10x10", # Ensure this matches the script name if run directly
)