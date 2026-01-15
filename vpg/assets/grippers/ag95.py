from dm_control import mjcf

class AG95:
    def __init__(self) -> None:
        self.mjcf_root = mjcf.from_path("vpg/assets/grippers/ag95.xml")

    @property
    def tcp_site_element(self):
        return self.mjcf_root.find("site", "gripper_center")
    
    @property
    def left_finger_element(self):
        return self.mjcf_root.find("body", "left_finger")
    
    @property
    def right_finger_element(self):
        return self.mjcf_root.find("body", "right_finger")



    @property
    def gpactuator_element(self):
        return self.mjcf_root.find("actuator","fingers_actuator")

