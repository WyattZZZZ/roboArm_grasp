class Util:
    @staticmethod
    def form_message(label):
        return label + "|" + "Rubber"

    @staticmethod
    def calculate_force(volume, density, factor, gravity=9.8):
        return volume * gravity * factor * density/2