class Util:
    @staticmethod
    def form_message(label):
        return label + "|" + "Rubber"

    @staticmethod
    def calculate_force(volume, density, factor, gravity=9.8):
        return volume * gravity * factor * density/2


class RoboArm:
    def __init__(self):
        self.gripper = None
        self.model = None

    def force_control(self, force):
        pass

    def grasp(self, num):
        pass
