import heapq
import numpy as np
from onpolicy.envs.mpe.core import World

class AStarPlanner:
    def __init__(self, world):
        world = World()
        self.world = world
        self.obstacle_list = [(dobstacle.state.p_pos, dobstacle.size) for dobstacle in world.dobstacles]

    def heuristic(self, a, b):
        return np.linalg.norm(a - b)

    def is_collision(self, point, radius):
        for (obs_pos, obs_size) in self.obstacle_list:
            if np.linalg.norm(point - obs_pos) <= radius + obs_size:
                return True
        return False

    def is_within_bounds(self, point):
        return -4 <= point[0] <= 4 and -4 <= point[1] <= 4

    def astar(self, start, goal):
        start = tuple(start)
        goal = tuple(goal)
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(np.array(start), np.array(goal))}

        while open_set:
            current = heapq.heappop(open_set)[1]

            if np.linalg.norm(np.array(current) - np.array(goal)) < 0.05:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                # path.append(start)
                path.reverse()
                # print(len(path))
                # print(f"guihua{path}")
                return [np.round(np.array(p),4) for p in path]

            for dx, dy in [(-0.05, 0), (0.05, 0), (0, -0.05), (0, 0.05)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if not self.is_within_bounds(neighbor):
                    continue

                tentative_g_score = g_score[current] + self.heuristic(np.array(current), np.array(neighbor))

                if self.is_collision(np.array(neighbor), 0.05):
                    continue

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(np.array(neighbor), np.array(goal))
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return []

if __name__ == "__main__":
    a = AStarPlanner(World)
    path = a.astar((0.5,0.5),(1,1))
    print(path)
    # def reconstruct_path(self, came_from, current):
    #     total_path = [current]
    #     while current in came_from:
    #         current = came_from[current]
    #         total_path.append(current)
    #     total_path.reverse()
    #     return total_path
    #
    # def get_neighbors(self, pos):
    #     neighbors = []
    #     for dx in [-1, 0, 1]:
    #         for dy in [-1, 0, 1]:
    #             if dx == 0 and dy == 0:
    #                 continue
    #             neighbor = (pos[0] + dx * self.world.dt, pos[1] + dy * self.world.dt)
    #             if self.is_valid(neighbor):
    #                 neighbors.append(neighbor)
    #     return neighbors
    #
    # def is_valid(self, pos):
    #     # Check if the position is within the world bounds and not colliding with obstacles
    #     x, y = pos
    #     if x < -1 or x > 1 or y < -1 or y > 1:
    #         return False
    #     for wall in self.world.walls:
    #         if wall.orient == 'H':
    #             if abs(y - wall.axis_pos) < wall.width / 2 and wall.endpoints[0] <= x <= wall.endpoints[1]:
    #                 return False
    #         elif wall.orient == 'V':
    #             if abs(x - wall.axis_pos) < wall.width / 2 and wall.endpoints[0] <= y <= wall.endpoints[1]:
    #                 return False
    #     return True
