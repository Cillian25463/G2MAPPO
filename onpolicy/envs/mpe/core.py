import numpy as np
import seaborn as sns
# from .a_star import AStarPlanner

# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None

# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None

# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None

# properties of wall entities
class Wall(object):
    def __init__(self, orient='H', axis_pos=0.0, endpoints=(-1, 1), width=0.1,
                 hard=True):
        # orientation: 'H'orizontal or 'V'ertical
        self.orient = orient
        # position along axis which wall lays on (y-axis for H, x-axis for V)
        self.axis_pos = axis_pos
        # endpoints of wall (x-coords for H, y-coords for V)
        self.endpoints = np.array(endpoints)
        # width of wall
        self.width = width
        # whether wall is impassable to all agents
        self.hard = hard
        # color of wall
        self.color = np.array([0.0, 0.0, 0.0])


# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # index among all entities (important to set for distance caching)
        self.i = 0
        # name
        self.name = ''
        # properties:
        self.size = 0.050
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # entity can pass through non-hard walls
        self.ghost = False
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.accel = None
        # state: including internal/mental state p_pos, p_vel
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0
        # commu channel
        self.channel = None

    @property
    def mass(self):
        return self.initial_mass

# properties of landmark entities
class Landmark(Entity):
    def __init__(self):
        super(Landmark, self).__init__()

class Dobstacle(Entity):
    def __init__(self):
        super(Dobstacle, self).__init__()
        self.path = []
        self.path_index = 0

# properties of agent entities
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # agent are adversary
        self.adversary = False
        # agent are dummy
        self.dummy = False
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # state: including communication state(communication utterance) c and internal/mental state p_pos, p_vel
        self.state = AgentState()
        # action: physical action u & communication action c
        self.action = Action()
        # script behavior to execute
        self.action_callback = None
        # zoe 20200420
        self.goal = None
        self.wallcol = False

# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        self.dobstacles = []
        self.walls = []
        # set a_star path
        self.paths = {}
        self.move_points = {}
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.1
        # physical damping（阻尼）
        self.damping = 0.25
        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3
        # cache distances between all agents (not calculated by default)
        self.cache_dists = False
        self.cached_dist_vect = None
        self.cached_dist_mag = None
        # zoe 20200420
        self.world_length = 25
        self.world_step = 0
        self.num_agents = 0
        self.num_landmarks = 0
        self.num_dobstacles = 0

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks + self.dobstacles

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    def calculate_distances(self):
        if self.cached_dist_vect is None:
            # initialize distance data structure
            self.cached_dist_vect = np.zeros((len(self.entities),
                                              len(self.entities),
                                              self.dim_p))
            # calculate minimum distance for a collision between all entities （size相加�?
            self.min_dists = np.zeros((len(self.entities), len(self.entities)))
            for ia, entity_a in enumerate(self.entities):
                for ib in range(ia + 1, len(self.entities)):
                    entity_b = self.entities[ib]
                    min_dist = entity_a.size + entity_b.size
                    self.min_dists[ia, ib] = min_dist
                    self.min_dists[ib, ia] = min_dist

        for ia, entity_a in enumerate(self.entities):
            for ib in range(ia + 1, len(self.entities)):
                entity_b = self.entities[ib]
                delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
                self.cached_dist_vect[ia, ib, :] = delta_pos
                self.cached_dist_vect[ib, ia, :] = -delta_pos

        self.cached_dist_mag = np.linalg.norm(self.cached_dist_vect, axis=2)

        self.cached_collisions = (self.cached_dist_mag <= self.min_dists)

    def assign_agent_colors(self):
        n_dummies = 0
        if hasattr(self.agents[0], 'dummy'):
            n_dummies = len([a for a in self.agents if a.dummy])
        n_adversaries = 0
        if hasattr(self.agents[0], 'adversary'):
            n_adversaries = len([a for a in self.agents if a.adversary])
        n_good_agents = len(self.agents) - n_adversaries - n_dummies
        # r g b
        dummy_colors = [(0.25, 0.75, 0.25)] * n_dummies
        # sns.color_palette("OrRd_d", n_adversaries)
        adv_colors = [(0.75, 0.25, 0.25)] * n_adversaries
        # sns.color_palette("GnBu_d", n_good_agents)
        good_colors = [(0.25, 0.25, 0.75)] * n_good_agents
        colors = dummy_colors + adv_colors + good_colors
        for color, agent in zip(colors, self.agents):
            agent.color = color

    # landmark color
    def assign_landmark_colors(self):
        for landmark in self.landmarks:
            landmark.color = np.array([0.0, 0.75, 0.0])

    def assign_dobstacles_colors(self):
        for dobstacle in self.dobstacles:
            dobstacle.color = np.array([0.75, 0.0, 0.0])
    # update state of the world
    def step(self):
        # zoe 20200420
        self.world_step += 1
        # set actions for scripted agents
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        # gather forces applied to entities
        p_force = [None] * len(self.entities)
        # apply agent physical controls
        p_force = self.apply_action_force(p_force)
        # apply environment forces
        p_force = self.apply_environment_force(p_force)
        # integrate physical state
        self.integrate_state(p_force)
        # for wall in self.walls:
        #     for agent in self.agents:
        #         force1 = self.get_wall_collision_force(agent, wall)
        #         if force1 is not None:
        #             agent.wallcol = True
        #             break

        for agent in self.agents:
            if not hasattr(agent, 'reached_goal'):
                agent.reached_goal = False
            # if not hasattr(agent, 'out_of_bounds'):
            #     agent.out_of_bounds = False  # 初始化 out_of_bounds
            if not agent.reached_goal:
                on_path = self.is_on_path(agent)
                agent.INPATH = on_path  # Add a flag to the agent indicating if it is on its path
                agent.num_move = len(self.move_points[agent.name])  # Update num_move with the length of move_points

                # 检查agent是否到达目标位置
                agent_index = int(agent.name.split()[-1])
                if agent_index < len(self.landmarks):
                    corresponding_landmark = self.landmarks[agent_index]
                    dist = np.sqrt(np.sum(np.square(agent.state.p_pos - corresponding_landmark.state.p_pos)))
                    if dist < 0.05:
                        agent.reached_goal = True
                        agent.state.p_vel = np.zeros(self.dim_p)
                        agent.state.p_pos = corresponding_landmark.state.p_pos  # 固定位置到目标点
            else:
                agent.state.p_vel = np.zeros(self.dim_p)
                agent.state.p_pos = self.landmarks[int(agent.name.split()[-1])].state.p_pos  # 固定位置到目标点

        # for agent in self.agents:
        #     if not hasattr(agent, 'reached_goal'):
        #         agent.reached_goal = False
        #     if not agent.reached_goal:
        #         on_path = self.is_on_path(agent)
        #         agent.INPATH = on_path  # Add a flag to the agent indicating if it is on its path
        #         agent.num_move = len(self.move_points[agent.name])  # Update num_move with the length of move_points
        #     else:
        #         agent.state.p_vel = np.zeros(self.world.dim_p)  # Stop the agent by setting its velocity to zero

        # 纠正agent
        for agent in self.agents:
            on_path = self.check_and_update_path(agent)
            agent.INPATH = on_path  # 添加一个标志，表示智能体是否在其路径上
            if agent.name in self.paths and len(self.paths[agent.name]) > 0:
                next_pos = self.paths[agent.name][0]
                direction = next_pos - agent.state.p_pos
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction /= norm
                    new_pos = agent.state.p_pos + direction * self.dt * agent.state.p_vel
                    if not self.is_dobstacles_collision(new_pos, agent.size):
                        agent.state.p_pos = new_pos

        for dobstacle in self.dobstacles:
            if dobstacle.movable:
                self.update_dynamic_obstacle_position(dobstacle)

    def update_dynamic_obstacle_position(self, dobstacle):
        """Update the position of a dynamic obstacle along its path."""
        path = dobstacle.path
        if not path or dobstacle.path_index >= len(path):
            dobstacle.state.p_vel = np.zeros(self.dim_p)  # Stop the obstacle
            return

        current_pos = dobstacle.state.p_pos
        next_pos = path[dobstacle.path_index]
        direction = next_pos - current_pos
        distance = np.linalg.norm(direction)

        if distance > 0.05:
            # Move in steps of 0.05
            step = 0.05 * direction / distance
            new_pos = current_pos + step
        else:
            # Move to the next position and update the index
            new_pos = next_pos
            dobstacle.path_index += 1

        dobstacle.state.p_pos = new_pos
        dobstacle.state.p_vel = (new_pos - current_pos) / 0.1
        # print(f"111{dobstacle.path}")
    # 判断是否在规划路径上并且计算movepoint的长度
    def is_on_path(self, agent):
        path = self.paths.get(agent.name, [])
        if not path:
            return False
        current_pos = np.round(agent.state.p_pos, 4)
        i = 0
        while i < len(path):
            point = path[i]
            if np.array_equal(current_pos, np.round(point, 4)):
                if agent.name not in self.move_points:
                    self.move_points[agent.name] = []
                self.move_points[agent.name].append(point)
                # Safely remove the point from the path
                path.pop(i)
                return True
            else:
                i += 1

        return False
        # for point in path:
        #     if np.array_equal(current_pos, np.round(point, 4)):
        #         # print(point)
        #         # path.remove(point)
        #         # shuswu = path.pop(0)
        #         # print(shuswu)
        #         # print(point,22)
        #         self.move_points[agent.name].append(point)
        #         return True
        # return False

        # update agent state
        # for agent in self.agents:
        #     self.update_agent_state(agent)
        # # calculate and store distances between all entities
        # if self.cache_dists:
        #     self.calculate_distances()

    # gather agent action forces
    def apply_action_force(self, p_force):
        # set applied forces
        for i, agent in enumerate(self.agents):
            if agent.movable:
                noise = np.random.randn(
                    *agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
                # force = mass * a * action + n
                p_force[i] = (
                    agent.mass * agent.accel if agent.accel is not None else agent.mass) * agent.action.u + noise
        return p_force

    # gather physical forces acting on entities
    def apply_environment_force(self, p_force):
        # simple (but inefficient) collision response
        for a, entity_a in enumerate(self.entities):
            for b, entity_b in enumerate(self.entities):
                if(b <= a):
                    continue
                [f_a, f_b] = self.get_entity_collision_force(a, b)
                if(f_a is not None):
                    if(p_force[a] is None):
                        p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a]
                if(f_b is not None):
                    if(p_force[b] is None):
                        p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]
            if entity_a.movable:
                for wall in self.walls:
                    wf = self.get_wall_collision_force(entity_a, wall)
                    if wf is not None:
                        if p_force[a] is None:
                            p_force[a] = 0.0
                        p_force[a] = p_force[a] + wf
        return p_force

    # def integrate_state(self, p_force):
    #     cam_range = 4  # Define the boundary range
    #     for i, entity in enumerate(self.entities):
    #         if isinstance(entity, Agent):  # 仅检查实体是否是智能体
    #             if not entity.movable:
    #                 continue
    #             if p_force[i] is not None:
    #                 entity.state.p_vel[0] += (p_force[i][0] / entity.mass) * self.dt
    #                 entity.state.p_vel[1] += (p_force[i][1] / entity.mass) * self.dt
    #
    #             # 更新位置
    #             new_pos = entity.state.p_pos + entity.state.p_vel * self.dt
    #
    #             # 使用cam_range进行检查和限制
    #             new_pos[0] = max(min(new_pos[0], cam_range), -cam_range)
    #             new_pos[1] = max(min(new_pos[1], cam_range), -cam_range)
    #
    #             entity.state.p_pos = new_pos

    def integrate_state(self, p_force):
        for i, entity in enumerate(self.entities):
            if isinstance(entity, Agent):  # 检查实体是否是智能体
                if not entity.movable:
                    continue
                if (p_force[i] is not None):
                    entity.state.p_vel[0] += (p_force[i][0] / entity.mass) * self.dt
                    entity.state.p_vel[1] += (p_force[i][1] / entity.mass) * self.dt
                # if entity.max_speed is not None:
                #     speed = np.sqrt(
                #         np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
                #     if speed > entity.max_speed:
                #         entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) +
                #                                                           np.square(entity.state.p_vel[1])) * entity.max_speed
                entity.state.p_pos += entity.state.p_vel * self.dt

        # for i, dobstacle in enumerate(self.dobstacles):
            # if dobstacle.movable:
            #     if p_force[dobstacle] is not None:
            #         dobstacle.state.p_vel[0] += (p_force[dobstacle][0] / dobstacle.mass) * self.dt
            #         dobstacle.state.p_vel[1] += (p_force[dobstacle][1] / dobstacle.mass) * self.dt
            #     if dobstacle.max_speed is not None:
            #         speed = np.sqrt(
            #             np.square(dobstacle.state.p_vel[0]) + np.square(dobstacle.state.p_vel[1]))
            #         if speed > dobstacle.max_speed:
            #             dobstacle.state.p_vel = dobstacle.state.p_vel / np.sqrt(np.square(dobstacle.state.p_vel[0]) +
            #                                                               np.square(dobstacle.state.p_vel[1])) * dobstacle.max_speed
            #     dobstacle.state.p_pos += dobstacle.state.p_vel * self.dt

    def update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * \
                agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise

    # get collision forces for any contact between two entities
    def get_entity_collision_force(self, ia, ib):
        entity_a = self.entities[ia]
        entity_b = self.entities[ib]
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None]  # not a collider
        if (not entity_a.movable) and (not entity_b.movable):
            return [None, None]  # neither entity moves
        if (entity_a is entity_b):
            return [None, None]  # don't collide against itself
        if self.cache_dists:
            delta_pos = self.cached_dist_vect[ia, ib]
            dist = self.cached_dist_mag[ia, ib]
            dist_min = self.min_dists[ia, ib]
        else:
            # compute actual distance between entities
            delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
            dist = np.sqrt(np.sum(np.square(delta_pos)))
            # minimum allowable distance
            dist_min = entity_a.size + entity_b.size
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min)/k)*k
        force = self.contact_force * delta_pos / dist * penetration
        if entity_a.movable and entity_b.movable:
            # consider mass in collisions
            force_ratio = entity_b.mass / entity_a.mass
            force_a = force_ratio * force
            force_b = -(1 / force_ratio) * force
        else:
            force_a = +force if entity_a.movable else None
            force_b = -force if entity_b.movable else None
        return [force_a, force_b]

    # get collision forces for contact between an entity and a wall
    def get_wall_collision_force(self, entity, wall):
        if entity.ghost and not wall.hard:
            return None  # ghost passes through soft walls
        if wall.orient == 'H':
            prll_dim = 0
            perp_dim = 1
        else:
            prll_dim = 1
            perp_dim = 0
        ent_pos = entity.state.p_pos
        if (ent_pos[prll_dim] < wall.endpoints[0] - entity.size or
                ent_pos[prll_dim] > wall.endpoints[1] + entity.size):
            return None  # entity is beyond endpoints of wall
        elif (ent_pos[prll_dim] < wall.endpoints[0] or
              ent_pos[prll_dim] > wall.endpoints[1]):
            # part of entity is beyond wall
            if ent_pos[prll_dim] < wall.endpoints[0]:
                dist_past_end = ent_pos[prll_dim] - wall.endpoints[0]
            else:
                dist_past_end = ent_pos[prll_dim] - wall.endpoints[1]
            theta = np.arcsin(dist_past_end / entity.size)
            dist_min = np.cos(theta) * entity.size + 0.5 * wall.width
        else:  # entire entity lies within bounds of wall
            theta = 0
            dist_past_end = 0
            dist_min = entity.size + 0.5 * wall.width

        # only need to calculate distance in relevant dim
        delta_pos = ent_pos[perp_dim] - wall.axis_pos
        dist = np.abs(delta_pos)
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min)/k)*k
        # force_mag = self.contact_force * delta_pos / dist * penetration
        force_mag = 1
        force = np.zeros(2)
        force[perp_dim] = np.cos(theta) * force_mag
        force[prll_dim] = np.sin(theta) * np.abs(force_mag)
        return force


    # 判断纠正过程中是否与障碍物碰撞
    def is_dobstacles_collision(self, pos, radius):
        for dobstacle in self.dobstacles:
            dist = np.linalg.norm(pos - dobstacle.state.p_pos)
            if dist <= radius + dobstacle.size:
                return True
        return False

    # 纠正的具体步骤
    def check_and_update_path(self, agent):
        on_path = False
        if agent.name in self.paths and len(self.paths[agent.name]) > 0:
            path = self.paths[agent.name]
            current_pos = agent.state.p_pos
            next_pos = path[0]
            if np.linalg.norm(current_pos - next_pos) < 0.1:  # Adjust threshold as needed
                on_path = True
                self.paths[agent.name].pop(0)  # Move to the next point in the path
            else:
                # Find the closest point on the path
                min_dist = float('inf')
                closest_point = None
                for point in path:
                    dist = np.linalg.norm(current_pos - point)
                    if dist < min_dist:
                        min_dist = dist
                        closest_point = point
                if closest_point is not None:
                    direction = closest_point - current_pos
                    norm = np.linalg.norm(direction)
                    if norm > 0:
                        direction /= norm
                        new_pos = current_pos + direction * 0.05  # Move agent slightly towards the path
                        if not self.is_dobstacles_collision(new_pos, agent.size):
                            agent.state.p_pos = new_pos
        return on_path