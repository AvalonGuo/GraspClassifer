
from dm_control import mjcf

class UR5E:
    def __init__(self):
        self.mjcf_root = mjcf.from_path("vpg/assets/robots/ur5e.xml")
        self._attachment_site = self.mjcf_root.find("site",self.attachment_site_name)
    @property
    def eef_site_name(self):
        return "eef_site"
    @property
    def attachment_site_name(self):
        return "attachment_site"
    
    @property
    def camera_element(self):
        return self.mjcf_root.find("camera","eyeinhand")



    @property
    def body_names(self):
        return ["shoulder_link","upper_arm_link","forearm_link","wrist_1_link","wrist_2_link","wrist_3_link"]
    @property
    def joint_names(self):
        return ["shoulder_pan_joint","shoulder_lift_joint","elbow_joint","wrist_1_joint","wrist_2_joint","wrist_3_joint"]
    

    def attach_tool(self,child, pos: list = [0, 0, 0], quat: list = [1, 0, 0, 0]) -> mjcf.Element:
        frame = self._attachment_site.attach(child)
        frame.pos = pos
        frame.quat = quat
        return frame

    
