
import pygame
import math
import random 
import queue 
from queue import PriorityQueue

#Window size
WIDTH = 800

#Creating window
WIN = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption("Pathfinding Visualizer")

#Set Colors
CLOSED_COLOR = (255, 128 , 114) #Salmon
OPEN_COLOR = (46, 139, 87) #SeaGreen
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
EMPTY_COLOR = (225, 225, 225) #light gray
BARRIER_COLOR = (0, 0, 0) #Black
PATH_COLOR = (90, 81, 190) #MediumSlateBlue  (123, 104, 238)
START_COLOR = (255, 165 ,0)
GREY = (128, 128, 128)
END_COLOR = (64, 224, 208)


def main(win, width):
    #Set number of Rows/Cols
    ROWS = 50
    
    grid = create_grid(ROWS, width)
    
    #Keep track whether the start and end nodes have been placed or not
    start = None 
    end = None
    
    run = True
    
    started = False #Keep track whether the visualizer has started

    maze =False #Doesnt allow random maze to be made if any nodes have already been placed

    while run:
        draw(win, grid, ROWS, width)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            
            if started: #If the visualizer has started, skip any inputs except QUIT
                continue

            if pygame.mouse.get_pressed()[0]: #Set Node States on left click
                mouse_pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(mouse_pos, ROWS, width)
                node = grid[row][col]
                if not start and node != end: #Create Start node on first click
                    start = node
                    start.make_start()

                elif not end and node != start: #Create End node on second click
                    end = node
                    end.make_end()

                elif node != end and node != start: #Create barriers after the second click
                    node.make_barrier()

            elif pygame.mouse.get_pressed()[2]: #Remove Node States on right click
                mouse_pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(mouse_pos, ROWS, width)
                node = grid[row][col]
                node.reset()
                if node == start:
                    start = None
                if node == end:
                    end = None

            if event.type == pygame.KEYDOWN: 
                if event.key == pygame.K_a and not started and start and end: #Start a* algorithm on A press
                    for row in grid:
                        for node in row:
                            node.update_neighbors(grid)
                    started = True
                    a_star_algo(lambda: draw(win, grid, ROWS, width), grid, start, end)
                    started = False

                if event.key == pygame.K_b and not started and start and end: #Start BFS algorithm on B press
                    for row in grid:
                        for node in row:
                            node.update_neighbors(grid)
                    started = True
                    bfs_algo(lambda: draw(win, grid, ROWS, width), grid, start, end)
                    started = False

                if event.key == pygame.K_d and not started and start and end: #Start DFS algorithm on D press
                    for row in grid:
                        for node in row:
                            node.update_dfs_neighbors(grid)
                    started = True
                    dfs_algo(lambda: draw(win, grid, ROWS, width), grid, start, end)
                    started = False
                

                if event.key == pygame.K_r and not started: #Reset entire grid on R press
                    for row in grid:
                        for node in row:
                            node.reset()
                            node.make_unvisited()
                            start = None
                            end = None
                            maze = False
                            
            
                if event.key == pygame.K_c and not started: #Clear all visualization nodes on C press
                    for row in grid:
                        for node in row:
                            node.clear_path()
                            

                if event.key == pygame.K_SPACE and not started and not start and not end and not maze: #generate random maze on Spacebar press
                    maze = True

                    started = True
                    
                    maze_generator(lambda: draw(win, grid, ROWS, width), grid, ROWS)
                    for rows in grid:
                        for node in rows:
                            if not node.been_visited:
                                node.make_barrier()

                            node.been_visited = False
                    started = False

    pygame.quit()

class Node:
    def __init__(self, row, col, width, total_rows):
        
        self.row = row
        self.col = col
        
        self.x = row * width
        self.y = col * width
        
        self.color = EMPTY_COLOR
        
        
        self.width = width
        self.total_rows = total_rows
        
        self.been_visited = False
        self.neighbors = []
        self.maze_neighbors = []
        self.available_neighbors = [] #Neighbors used in the maze generator
        self.dfs_neighbors = []

    def been_visited(self): #Checks whether the node has been visited by another node already
        return self.been_visited

    def get_pos(self): 
        return self.row, self.col
    
    
    #Functions to check the status of the node
    def is_closed(self):
        return self.color == CLOSED_COLOR
    
    def is_open(self):
        
        return self.color == OPEN_COLOR
        
    def is_barrier(self):
        return self.color == BARRIER_COLOR

    def is_start(self):
        return self.color == START_COLOR
    
    def is_end(self):
        return self.color == END_COLOR

    def is_available(self, grid):
        #Used in the maze generator
        
        #Counts how many of its neighbors have been visited by other nodes already
        count =0 
        
        if self.row < self.total_rows - 1 and grid[self.row + 1][self.col].been_visited:
            count += 1
        if self.row > 0 and grid[self.row - 1][self.col].been_visited:
            count += 1
        if self.col < self.total_rows - 1 and grid[self.row][self.col + 1].been_visited:
            count += 1
        if self.col > 0 and grid[self.row][self.col - 1].been_visited:
            count += 1
        
        
        #If 3 or 4 of its neighbors have not been visited, this node is available
        if count <2:
            return True
        return False
        #one of the neighbors will always be the current node so count will always be at least 1
        #this means that the other 3 neighbors must be available
        #this allows a single file path without any lumping
    
    
    
    
    def make_unvisited(self):
        self.been_visited = False

    def make_visited(self):
        
        self.been_visited = True

    def make_start(self):
        self.color = START_COLOR

    def reset(self):
        self.color = EMPTY_COLOR

    def make_closed(self):
        self.color = CLOSED_COLOR

    def make_open(self):
        if not self.is_end() and not self.is_start():
            self.color = OPEN_COLOR
        
        
    
    def make_barrier(self):
        self.color = BARRIER_COLOR

    def make_end(self):
         self.color = END_COLOR
        
    def make_path(self):
        self.color = PATH_COLOR

    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))
        
        
    

    def update_neighbors(self, grid): #appends list of neighbors that arent barriers to each node
    #Used in A* Algo, BFS Algo
        self.neighbors = [] 
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier(): #DOWN
            self.neighbors.append(grid[self.row + 1][self.col])

        if self.row > 0 and not grid[self.row - 1][self.col].is_barrier(): #UP
            self.neighbors.append(grid[self.row - 1][self.col])

        if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrier(): #RIGHT
            self.neighbors.append(grid[self.row][self.col + 1])

        if self.col > 0 and not grid[self.row][self.col - 1].is_barrier(): #LEFT
            self.neighbors.append(grid[self.row][self.col - 1])

    def update_maze_neighbors(self, grid): # appends list of neighbors that havent been visited yet
    #Used for the maze generators
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].been_visited: #DOWN
            self.maze_neighbors.append(grid[self.row + 1][self.col])

        if self.row > 0 and not grid[self.row - 1][self.col].been_visited: #UP
            self.maze_neighbors.append(grid[self.row - 1][self.col])

        if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].been_visited: #RIGHT
            self.maze_neighbors.append(grid[self.row][self.col + 1])

        if self.col > 0 and not grid[self.row][self.col - 1].been_visited: #LEFT
            self.maze_neighbors.append(grid[self.row][self.col - 1])

    def update_dfs_neighbors(self, grid): #dfs neighbors are not closed nor barriers nor start nodes
    #used for the dfs algo
        if self.col > 0 and not grid[self.row][self.col - 1].is_closed() and not grid[self.row][self.col - 1].is_barrier() and not grid[self.row][self.col - 1].is_start(): #LEFT
            self.dfs_neighbors.append(grid[self.row][self.col - 1])

        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_closed() and not grid[self.row + 1][self.col].is_barrier() and not grid[self.row + 1][self.col].is_start() : #DOWN
            self.dfs_neighbors.append(grid[self.row + 1][self.col])

        if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_closed() and not grid[self.row][self.col + 1].is_barrier() and not grid[self.row][self.col + 1].is_start(): #RIGHT
            self.dfs_neighbors.append(grid[self.row][self.col + 1])

        if self.row > 0 and not grid[self.row - 1][self.col].is_closed() and not grid[self.row - 1][self.col].is_barrier() and not grid[self.row - 1][self.col].is_start(): #UP
            self.dfs_neighbors.append(grid[self.row - 1][self.col])
    
    def clear_path(self): #Reset all visualization nodes
        if self.color == CLOSED_COLOR or self.color == OPEN_COLOR or self.color == PATH_COLOR:
            self.reset()


    def __lt__(self, other):
        return False

def maze_generator(draw, grid, rows): #First in first out algorithm that searches for nodes which have not been visited nor have 2 neighbors that have been visited
    count = 0
    start_x = random.randrange(rows)
    start_y = random.randrange(rows)
    start_node = grid[start_x][start_y] #random starting location

    open_set = queue.Queue(0)
    open_set.put((count, start_node))
    open_set_hash = {start_node}
    
    backtracking_queue = queue.Queue(0)
    backtracking_queue.put((0, start_node))
    backtracking_queue_hash = {start_node}
    backtracking_count = 0
    
    while not open_set.empty():
        for event in pygame.event.get(): 
            if event.type == pygame.QUIT:
                pygame.quit()
        
        current = open_set.get()[1] #get the node element from the queue
        current.make_visited()

        current.maze_neighbors = [] #reset neighbors
        current.available_neighbors = [] #reset neighbors

        current.update_maze_neighbors(grid)
        count += 1

        
        
        for neighbor in current.maze_neighbors: #update available neighbors
            if neighbor.is_available(grid):
                current.available_neighbors.append(neighbor)

        if(len(current.available_neighbors) != 0): 
            #if there is an available neighbor, randomly select one
            
            x = random.randrange(len(current.available_neighbors))
            next_node = current.available_neighbors[x]
            open_set.put((count, next_node))
            open_set_hash.add(next_node)
            open_set_hash.remove(current)
            
           
            if (len(current.available_neighbors) == 2): 
                #If there is 2 neighbors add the unselected node to the backtracking queue
                if x == 0:
                    backtracking_queue.put((backtracking_count, current.available_neighbors[1]))
                    backtracking_queue_hash.add(current.available_neighbors[1])
                    backtracking_count += 1
                elif x == 1:
                    backtracking_queue.put((backtracking_count, current.available_neighbors[0]))
                    backtracking_queue_hash.add(current.available_neighbors[0])
                    backtracking_count += 1

            elif (len(current.available_neighbors) == 3):
                #if there are 3 neighbors add the 2 unselected nodes to the backtracking queue
                if (x == 0):
                    backtracking_queue.put((backtracking_count, current.available_neighbors[1]))
                    backtracking_queue_hash.add(current.available_neighbors[1])
                    backtracking_queue.put((backtracking_count, current.available_neighbors[2]))
                    backtracking_queue_hash.add(current.available_neighbors[2])
                    backtracking_count += 1
                elif (x == 1):
                    backtracking_queue.put((backtracking_count, current.available_neighbors[0]))
                    backtracking_queue_hash.add(current.available_neighbors[0])
                    backtracking_queue.put((backtracking_count, current.available_neighbors[2]))
                    backtracking_queue_hash.add(current.available_neighbors[2])
                    backtracking_count += 1
                elif (x == 2):
                    backtracking_queue.put((backtracking_count, current.available_neighbors[0]))
                    backtracking_queue_hash.add(current.available_neighbors[0])
                    backtracking_queue.put((backtracking_count, current.available_neighbors[1]))
                    backtracking_queue_hash.add(current.available_neighbors[1])
                    backtracking_count += 1

            
            current.reset()
           
        else: #if no open nodes, look back at passed open nodes
            
            while not backtracking_queue.empty():
                
                available_node = backtracking_queue.get()[1]
                
                available_node.maze_neighbors = []
                available_node.available_neighbors = []
                available_node.update_maze_neighbors(grid)
                count = 0
                
                for neighbor in available_node.maze_neighbors: 
                    #updates available neighbors for potential next node
                    
                    if neighbor.is_available(grid):
                        available_node.available_neighbors.append(neighbor)

                
               
                if available_node.is_available(grid): 
                    #if the node is still available set it to be the next node
                    #remove it from the backtracking, and exit the while loop
                    
                    open_set.put((count, available_node))
                    open_set_hash.add(available_node)
                    open_set_hash.remove(current)
                    current.reset()
                    count += 1
                    backtracking_queue_hash.remove(available_node)
                    break
        
    return False

def h(p1, p2): #Returns estimated distance, Manhattan distance
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)

def reconstruct_path(came_from, current, draw): #draws the shortest path
    while current in came_from:
        current = came_from[current]
        current.make_path()
        draw()

def a_star_algo(draw, grid, start, end):
    #Algorithm which takes a heuristic value of an estimated distance to the end node (Manhattan distance)
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    came_from = {} #Keeps track of path
    g_score = {node: float("inf") for row in grid for node in row} #Set default g scores to infinity
    g_score[start] = 0 
    f_score = {node: float("inf") for row in grid for node in row}#Set default f scores to infinity
    f_score[start] = h(start.get_pos(), end.get_pos())

    open_set_hash = {start}

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current = open_set.get()[2]
        open_set_hash.remove(current)

        if current == end: #Create Path when end node is found
            reconstruct_path(came_from, end, draw)
            end.make_end() 
            start.make_start()            
            return True
        
        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1 

            if temp_g_score < g_score[neighbor]: #if this path has a better g_score, override the path and update the scores
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + h(neighbor.get_pos(), end.get_pos())
                if neighbor not in open_set_hash: #if this is a new path, add it to the queue
                    count += 1
                    open_set.put((f_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor) 
                    neighbor.make_open()
        
        draw()

        if current != start:
            current.make_closed()
    return False

def bfs_algo(draw, grid, start, end): 
    #First in first out algorithm where all currently open paths are equal length
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    came_from = {}
    open_set_hash = {start}

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current = open_set.get()[2]
        open_set_hash.remove(current)


        if current == end: #When the end is found, reconstrunct the path
            reconstruct_path(came_from, end, draw)
            end.make_end()
            start.make_start()
            return True

        for neighbor in current.neighbors: 
            #Adds each neighbor that isnt closed or the start the node into the queue
            if not neighbor.is_closed() and not neighbor.is_start():
                came_from[neighbor] = current
                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((0, count, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()

        draw()

        if current != start:
            current.make_closed()
    return False

def dfs_algo(draw, grid, start, end): 
    #Last in First out algorithm that backtracks to the lastest available node
    count = 0
    open_set = queue.Queue(0)
    open_set.put((count, start))
    open_set_hash = {start}
    
    backtracking_queue = queue.LifoQueue(0)
    backtracking_queue.put((0, start))
    backtracking_queue_hash = {start}
    backtracking_count = 0

    came_from = {}
    
    while not open_set.empty():
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        
        current = open_set.get()[1]
        if current == end:
            reconstruct_path(came_from, end, draw)
            end.make_end()
            start.make_start()
            return True

        if current != start:
            current.make_closed()
        
        
        current.dfs_neighbors = []
        current.update_dfs_neighbors(grid)
        count += 1
        neighbors_length = len(current.dfs_neighbors)
        
        if(neighbors_length != 0): #if there is an open node, select the first
            
            
            next_node = current.dfs_neighbors[0]
            open_set.put((count, next_node))
            open_set_hash.add(next_node)
            open_set_hash.remove(current)

            came_from[next_node] = current

            backtracking_queue.put((backtracking_count, current))
            backtracking_queue_hash.add(current)
            backtracking_count += 1
            
            for i in range(len(current.dfs_neighbors) -1):
                current.dfs_neighbors[i+1].make_open()
                

            
        

        else: #if no open nodes, look back at passed open nodes
            
            while not backtracking_queue.empty():
                
                available_node = backtracking_queue.get()[1]
                 
                
                available_node.dfs_neighbors = []
                available_node.update_dfs_neighbors(grid)
                neighbors_length_2 = len(available_node.dfs_neighbors)
                count = 0
                
                if neighbors_length_2 != 0:
                    open_set.put((count, available_node))
                    open_set_hash.add(available_node)
                    open_set_hash.remove(current)
                    
                    count += 1
                    backtracking_queue_hash.remove(available_node) 
                    break



        draw()

        
    
    return False
    
    

def create_grid(rows, width):
    grid =[]
    gap = width // rows
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            node = Node(i, j, gap, rows)
            grid[i].append(node)

    return grid

def draw_grid(win, rows, width):
    gap = width // rows
    for i in range(rows):
        pygame.draw.line(win, GREY, (0, i * gap), (width, i * gap))
        for j in range(rows):
            pygame.draw.line(win, GREY, (j * gap, 0), (j * gap, width))

def draw(win, grid, rows, width):
    win.fill(EMPTY_COLOR)

    for row in grid:
        for node in row:
            node.draw(win)

    draw_grid(win, rows, width)
    pygame.display.update()

def get_clicked_pos(pos, rows, width):
    gap = width // rows
    y, x = pos

    row = y // gap
    col = x // gap

    return row, col

main(WIN, WIDTH)