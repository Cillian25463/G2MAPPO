import numpy as np
from onpolicy.envs.mpe.core import World, Agent, Landmark, Dobstacle, Wall
from onpolicy.envs.mpe.scenario import BaseScenario
from onpolicy.envs.mpe.environment import MultiAgentEnv
from onpolicy.envs.mpe.a_star import AStarPlanner



class Scenario(BaseScenario):
    def make_world(self, args):
        world = World()
        world.world_length = args.episode_length
        # set any world properties first
        world.dim_c = 2
        world.num_agents = args.num_agents
        world.num_landmarks = args.num_landmarks  # 3
        world.num_dobstacles = 5
        world.collaborative = True
        # self.count = 0
        # add agents
        world.agents = [Agent() for i in range(world.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.05
        # add landmarks
        world.landmarks = [Landmark() for i in range(world.num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False

        world.dobstacles = [Dobstacle() for i in range(world.num_dobstacles)]
        for j, dobstacle in enumerate(world.dobstacles):
            dobstacle.name = 'dobstacle %d' % i
            dobstacle.collide = True
            if j < 3:
                dobstacle.movable = False
            else:
                dobstacle.movable = True


        # Add walls
        world.walls = [
            Wall(orient='H', axis_pos=4.0, endpoints=(-4, 4), width=0.1, hard=True),
            Wall(orient='V', axis_pos=-4.0, endpoints=(-4, 4), width=0.1, hard=True),
            Wall(orient='H', axis_pos=-4.0, endpoints=(-4, 4), width=0.1, hard=True),
            Wall(orient='V', axis_pos=4.0, endpoints=(-4, 4), width=0.1, hard=True)
            # Wall(orient='V', axis_pos=1.0, endpoints=(1.2, -1.2), width=0.05, hard=True)
            # Add more walls as needed
        ]
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # self.count += 1
        #
        # if self.count % 200 == 0:
            np.random.seed(42)
            # random properties for agents
            world.assign_agent_colors()

            world.assign_landmark_colors()

            world.assign_dobstacles_colors()

            # set random initial states
            for agent in world.agents:
                agent.state.p_pos = np.round(np.random.uniform(-2, 1, world.dim_p),4)
                agent.state.p_vel = np.zeros(world.dim_p)
                agent.state.c = np.zeros(world.dim_c)
                agent.reached_goal = False  # 初始化reached_goal标志
                agent.out_of_bounds = False  # 初始化out_of_bounds标志
            for i, landmark in enumerate(world.landmarks):
                landmark.state.p_pos = np.round(0.8 * np.random.uniform(2, -1, world.dim_p),4)
                landmark.state.p_vel = np.zeros(world.dim_p)

            for j, dobstacle in enumerate(world.dobstacles):
                if j < 3:
                    dobstacle.state.p_pos = np.round(np.random.uniform(-2 ,+2, 2),4)  # 静止障碍物的随机位置
                    dobstacle.state.p_vel = np.zeros(2)  # 静止障碍物的初始速度为零
                else:
                    dobstacle.state.p_pos = np.round(np.random.uniform(-2, +2, 2),4)  # 动态障碍物的随机位置
                    dobstacle.state.p_vel = np.random.uniform(-1, 1, 2)  # 动态障碍物的随机初始速度

                if dobstacle.movable:
                    # Define start and end points for dynamic obstacles
                    dobstacle.start_point = dobstacle.state.p_pos
                    dobstacle.end_point = np.round(np.random.uniform(-2, +2, world.dim_p), 4)
                    dobstacle.path = self.create_path(dobstacle.start_point, dobstacle.end_point)

            # Instantiate the AStarPlanner with the current world
            planner = AStarPlanner(world)
            # world.paths
            world.move_points = {}

            # Calculate and print paths from each agent to the corresponding landmark
            for i, agent in enumerate(world.agents):
                agent.name = f'agent {i}'  # Ensure agent.name is set to a unique identifier
                if i < len(world.landmarks):
                    start = agent.state.p_pos
                    goal = world.landmarks[i].state.p_pos
                    # print(start,goal)
                    path = planner.astar(start, goal)
                    world.paths[agent.name] = path  # Store path as a list for each agent in the paths dictionary
                    world.move_points[agent.name] = []
                    # print(f'Path for {agent.name} to landmark {world.landmarks[i].name}: {path}')

            for j, dobstacle in enumerate(world.dobstacles):
                dobstacle.state.p_pos = np.round(np.random.uniform(-2, 2, 2), 4)
                dobstacle.path_index = 0
                if dobstacle.movable:
                    agent_index = j % len(world.agents)  # Ensure each obstacle follows a different agent's path
                    agent_path = world.paths[f'agent {agent_index}']

                    if len(agent_path) > 4:
                        buffer_distance = 0.2
                        valid_points = [p for p in agent_path if
                                        np.linalg.norm(p - agent_path[0]) > buffer_distance and
                                        np.linalg.norm(p - agent_path[-1]) > buffer_distance]
                        if len(valid_points) >= 2:
                            dobstacle.path = valid_points
                        else:
                            dobstacle.path = agent_path[1:-1]
                    else:
                        dobstacle.path = agent_path

                    if dobstacle.path:
                        dobstacle.path.reverse()  # Reverse the path to go in the opposite direction of the agent
                        dobstacle.state.p_pos = dobstacle.path[0]
                        dobstacle.state.p_vel = (dobstacle.path[1] - dobstacle.path[0]) / 0.1



            return world.paths

    def create_path(self, start, end, num_points=20, buffer_distance=0.1):
        """Create a linear path from start to end with num_points, excluding points near the start and end."""
        path = []
        for i in range(num_points):
            point = start + (end - start) * i / (num_points - 1)
            if np.linalg.norm(point - start) > buffer_distance and np.linalg.norm(point - end) > buffer_distance:
                path.append(point)
        return path

    # def benchmark_data(self, agent, world):
    #     rew = 0
    #     collisions = 0
    #     occupied_landmarks = 0
    #     occupied_dobstacles = 0
    #     min_dists = 0
    #     # Check if agent is on its planned path
    #     if planned_path:
    #         if tuple(np.round(agent.state.p_pos, 1)) in planned_path:
    #             rew += 20.0  # Reward for being on the planned path
    #         else:
    #             rew -= 5.0  # Penalty for deviating from the planned path
    #     for l in world.landmarks:
    #         dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos)))
    #                  for a in world.agents]
    #         min_dists += min(dists)
    # #         rew -= min(dists)
    # #         if min(dists) < 0.1:
    # #             occupied_landmarks += 1
    #
    #     for d in world.dobstacles:
    #         dists = [np.sqrt(np.sum(np.square(a.state.p_pos - d.state.p_pos)))
    #                  for a in world.agents]
    #         if min(dists) < 0.2:
    #             rew -= 100
    #             occupied_dobstacles += 1
    #
    #     if agent.collide:
    #         for a in world.agents:
    #             if self.is_collision(a, agent):
    #                 rew -= 1
    #                 collisions += 1
    #         for d in world.dobstacles:
    #             if self.is_collision(a, d):
    #                 rew -= 1
    #                 collisions += 1
    #     return (rew, collisions, min_dists, occupied_landmarks, occupied_dobstacles)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # def reward(self, agent, world):
    #     # 计算已经到达landmark的agent数量
    #     reached_agents = sum(1 for a in world.agents if hasattr(a, 'reached_goal') and a.reached_goal)
    #
    #     # 设置基础移动成本和额外的移动成本因子
    #     base_move_cost = 0.1
    #     additional_move_cost_per_agent = 0.1
    #
    #     # 初始化奖励
    #     rew = 0
    #
    #     # 获取智能体的索引
    #     agent_index = int(agent.name.split()[-1])
    #
    #     # 计算智能体与其对应landmark的距离
    #     if agent_index < len(world.landmarks):
    #         corresponding_landmark = world.landmarks[agent_index]
    #         dist = np.sqrt(np.sum(np.square(agent.state.p_pos - corresponding_landmark.state.p_pos)))
    #
    #         # 奖励接近landmark的agent
    #         if dist < 0.05:
    #             rew += 20
    #             agent.reached_goal = True
    #             agent.state.p_pos = corresponding_landmark.state.p_pos  # 固定位置到目标点
    #         elif dist < 0.1:
    #             rew += 10
    #         elif dist < 0.2:
    #             rew += 5
    #
    #         # 基于距离的奖励
    #         rew -= dist * 0.1
    #
    #     # 惩罚与其他agent或动态障碍物碰撞的agent
    #     if agent.collide:
    #         for a in world.agents:
    #             if self.is_collision(a, agent):
    #                 rew -= 15
    #         for d in world.dobstacles:
    #             if self.is_collision(d, agent):
    #                 rew -= 15
    #
    #     # 惩罚与墙壁碰撞的agent
    #     # if agent.wallcol:
    #     #     rew -= 20
    #     #     agent.wallcol = False
    #
    #     if agent.out_of_bounds:
    #         rew -= 50  # 超出地图范围的惩罚
    #
    #     # 增加移动成本
    #     move_cost = base_move_cost + reached_agents * additional_move_cost_per_agent
    #     rew -= move_cost
    #
    #     # 计算共享奖励
    #     shared_reward = 0
    #     for l in world.landmarks:
    #         dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos)))
    #                  for a in world.agents]
    #         shared_reward -= min(dists)
    #         if min(dists) < 0.005:
    #             shared_reward += 50
    #
    #     # 综合总奖励
    #     alpha = 0.5  # 共享奖励的权重，可以根据需要调整
    #     total_reward = alpha * shared_reward + (1 - alpha) * rew
    #
    #     return total_reward

    def reward(self, agent, world):
        # Initialize the reward
        rew = 0

        # Step penalty
        rew -= 0.1

        # Penalty for collisions with other agents
        if agent.collide:
            for a in world.agents:
                if a is not agent and self.is_collision(a, agent):
                    rew -= 5

        # Penalty for collisions with dynamic obstacles
        for dobstacle in world.dobstacles:
            if self.is_collision(agent, dobstacle):
                rew -= 5


        x_min, x_max, y_min, y_max = -4, 4, -4, 4

        # 检查 agent 是否超出地图边界
        pos = agent.state.p_pos
        if pos[0] < x_min or pos[0] > x_max or pos[1] < y_min or pos[1] > y_max:
            rew -= 30

        # Penalty for deviating from the planned path
        if agent.name in world.paths:
            path = world.paths[agent.name]
            if path:
                current_pos = np.array(agent.state.p_pos)
                distances = [np.linalg.norm(current_pos - np.array(point)) for point in path]
                min_distance = min(distances)
                rew -= min_distance

        # Reward for reaching the corresponding landmark
        agent_index = int(agent.name.split()[-1])
        if agent_index < len(world.landmarks):
            corresponding_landmark = world.landmarks[agent_index]
            if np.linalg.norm(agent.state.p_pos - corresponding_landmark.state.p_pos) < 0.1:
                rew += 30
                agent.reached_goal = True  # Mark the goal as reached

        # Penalty for movement oscillation
        if hasattr(agent, 'previous_positions'):
            if len(agent.previous_positions) >= 3:
                if np.array_equal(agent.previous_positions[-1], agent.state.p_pos) and \
                   np.array_equal(agent.previous_positions[-3], agent.state.p_pos):
                    rew -= 0.3
            agent.previous_positions.append(agent.state.p_pos)
            if len(agent.previous_positions) > 10:
                agent.previous_positions.pop(0)
        else:
            agent.previous_positions = [agent.state.p_pos]


        return rew


    # def reward(self, agent, world):
    #     # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
    #     rew = 0
    #
    #     if agent.INPATH:
    #         rew += 2
    #         # rew += 0.25 * len(world.move_points.get(agent.name, []))  # 如果在规划路径上，奖励5 * move_points的长度
    #     else:
    #         rew -= 1  # 如果不在，惩罚1分
    #
    #     # 获取智能体的索引
    #     agent_index = int(agent.name.split()[-1])
    #
    #     # 计算智能体与其对应landmark的距离
    #     if agent_index < len(world.landmarks):
    #         corresponding_landmark = world.landmarks[agent_index]
    #         dist = np.sqrt(np.sum(np.square(agent.state.p_pos - corresponding_landmark.state.p_pos)))
    #         if dist < 0.0005:
    #                 rew += 20
    #
    #     # for l in world.landmarks:
    #     #     dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos)))
    #     #              for a in world.agents]
    #     #     rew -= min(dists)
    #     #     if min(dists) < 0.005:
    #     #         rew += 20
    #
    #     cam_range = 4
    #     if any(abs(coord) > cam_range for coord in agent.state.p_pos):
    #         rew -= 20  # 如果超出范围，惩罚1分
    #
    #     if agent.collide:
    #         for a in world.agents:
    #             if self.is_collision(a, agent):
    #                 rew -= 15
    #         for d in world.dobstacles:
    #             if self.is_collision(d, agent):
    #                 rew -= 15
    #     return rew


    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        for entity in world.dobstacles:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        for entity in world.dobstacles:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)

    # # 全局路径加上部分观测
    # def observation(self, agent, world):
    #     # 定义局部观测范围
    #     local_view_range = 1.0  # 定义agent能观测到的范围
    #
    #     def in_view_range(pos):
    #         return np.linalg.norm(agent.state.p_pos - pos) <= local_view_range
    #
    #     # 获取所有实体在该agent参考框架中的位置
    #     entity_pos = []
    #     for entity in world.landmarks:
    #         if in_view_range(entity.state.p_pos):
    #             entity_pos.append(entity.state.p_pos - agent.state.p_pos)
    #     for entity in world.dobstacles:
    #         if in_view_range(entity.state.p_pos):
    #             entity_pos.append(entity.state.p_pos - agent.state.p_pos)
    #
    #     # 获取实体的颜色
    #     entity_color = []
    #     for entity in world.landmarks:
    #         entity_color.append(entity.color)
    #     for entity in world.dobstacles:
    #         entity_color.append(entity.color)
    #
    #     # 获取其他agent的通信信息和位置
    #     comm = []
    #     other_pos = []
    #     for other in world.agents:
    #         if other is agent:
    #             continue
    #         comm.append(other.state.c)
    #         if in_view_range(other.state.p_pos):
    #             other_pos.append(other.state.p_pos - agent.state.p_pos)
    #
    #     # 获取全局路径信息
    #     path_info = []
    #     # if agent.name in world.paths:
    #     #     agent_path = world.paths[agent.name]
    #     #     for point in agent_path:
    #     #         if in_view_range(point):
    #     #             path_info.append(point - agent.state.p_pos)
    #
    #     # 将所有观测信息合并成一个向量
    #     return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm + path_info)

    只有部分观测的
    def observation(self, agent, world):
        local_view_range = 1.0  # Define the range within which the agent can observe

        def in_view_range(pos):
            return np.linalg.norm(agent.state.p_pos - pos) <= local_view_range

        # Local observation of entities (landmarks and dynamic obstacles)
        entity_pos = [entity.state.p_pos - agent.state.p_pos for entity in world.landmarks + world.dobstacles if
                      in_view_range(entity.state.p_pos)]

        # Local observation of walls
        wall_info = []
        for wall in world.walls:
            if wall.orient == 'H':
                if abs(agent.state.p_pos[1] - wall.axis_pos) <= local_view_range:
                    wall_info.append([wall.axis_pos, wall.endpoints[0], wall.endpoints[1], wall.width])
            else:
                if abs(agent.state.p_pos[0] - wall.axis_pos) <= local_view_range:
                    wall_info.append([wall.axis_pos, wall.endpoints[0], wall.endpoints[1], wall.width])

        # Part of global path information
        path_info = []
        if agent.name in world.paths:
            agent_path = world.paths[agent.name]
            path_info = [point - agent.state.p_pos for point in agent_path if in_view_range(point)]

        # Communication and positions of other agents
        comm = [other.state.c for other in world.agents if other is not agent]
        other_pos = [other.state.p_pos - agent.state.p_pos for other in world.agents if
                     other is not agent and in_view_range(other.state.p_pos)]

        return np.concatenate(
            [agent.state.p_vel, agent.state.p_pos] + entity_pos + other_pos + comm + wall_info + path_info)







