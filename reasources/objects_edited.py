from enum import Enum

movement_threshold = 60


class Direction(Enum):
    UP = 1
    DOWN = 2


class Object:
    history = []

    def __init__(self, id, x, y, timeout):
        self.id = id
        self.x = x
        self.y = y
        self.history = []
        self.done = False
        self.timeout = timeout
        self.direction = None
        self.pulse = 0
        self.history.append([self.x, self.y])

    def get_id(self):  # For the ID
        return self.id

    def in_contour(self, contour_x, contour_y, contour_width, contour_height):
        return abs(contour_x - self.x) <= contour_width and abs(contour_y - self.y) <= contour_height

    def update_coordinates(self, new_x, new_y):
        # print(abs(self.history[-1][1]-yn))
        if (abs(self.history[-1][1]-new_y) > movement_threshold) or (abs(self.history[-1][0]-new_x) > movement_threshold):
            # print("outside")
            return
        self.pulse = 0
        self.x = new_x
        self.y = new_y
        self.history.append([self.x, self.y])

    def set_done(self):
        self.done = True

    def is_done(self):
        return self.done

    def going_up(self, up_line):
        if self.is_done():
            return False
        if len(self.history) >= 2:
            if self.history[-1][1] < up_line <= self.history[-2][1]:
                self.direction = Direction.UP
                self.done = True
                return True
            else:
                return False
        else:
            return False

    def going_down(self, down_line):
        if self.is_done():
            return False
        if len(self.history) >= 2:
            if self.history[-1][1] > down_line >= self.history[-2][1]:
                self.direction = Direction.DOWN
                self.done = True
                return True
            else:
                return False
        else:
            return False

    def age(self):
        self.pulse += 1

    def check_health(self):
        if self.done:
            return
        if self.pulse > self.timeout:
            self.set_done()
            print("killed", self.id)




