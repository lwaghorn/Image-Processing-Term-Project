


class Car:
    tracks = []

    def __init__(self, i, xi, yi, max_age):
        self.i = i
        self.x = xi
        self.y = yi
        self.tracks = []
        self.done = False
        self.state = '0'
        self.age = 0
        self.max_age = max_age
        self.dir = None

    def get_tracks(self):
        return self.tracks

    def get_id(self):  # For the ID
        return self.i

    def get_direction(self):
        return self.dir

    def in_contour(self, contour_x, contour_y, contour_width, contour_height):
        return abs(contour_x - self.x) <= contour_width and abs(contour_y - self.y) <= contour_height

    def update_coordinates(self, xn, yn):
        self.age = 0
        self.tracks.append([self.x, self.y])
        self.x = xn
        self.y = yn

    def set_done(self):
        self.done = True

    def is_done(self):
        return self.done

    def going_up(self, up_line):
        if len(self.tracks) >= 2:
            if self.tracks[-1][1] < up_line <= self.tracks[-2][1]:
                self.dir = 'up'
                self.done = True
                return True
            else:
                return False
        else:
            return False

    def going_down(self, down_line):
        if len(self.tracks) >= 2:
            if self.tracks[-1][1] > down_line >= self.tracks[-2][1]:
                self.dir = 'down'
                self.done = True
                return True
            else:
                return False
        else:
            return False


