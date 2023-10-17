from datetime import datetime

class Session:
    def __init__(self, cur_model="tiny", epochs=10):
        # CONFIGS
        self.CUR_MODEL= cur_model
        self.BATCH_SIZE= 32
        self.NUM_WORKERS= 32
        self.EPOCHS= epochs

        # CONSTANTS
        self.CLASS_MAPPING = {
            "0": "Atelectasis", 
            "1": "Cardiomegaly", 
            "2": "Consolidation", 
            "3": "Edema", 
            "4": "Effusion", 
            "5": "Emphysema", 
            "6": "Fibrosis", 
            "7": "Hernia", 
            "8": "Infiltration", 
            "9": "Mass", 
            "10": "Nodule", 
            "11": "Pleural_Thickening", 
            "12": "Pneumonia", 
            "13": "Pneumothorax", 
            "14": "No Finding"
        }
        self.MODELS = {
            "tiny": "convnext_tiny.fb_in22k",
            "small": "convnext_small.fb_in22k",
            "base": "convnext_base.fb_in22k",
            "large": "convnext_large.fb_in22k",
            "xlarge": "convnext_xlarge.fb_in22k"
        }
        self.NUM_LABELS = 15

        # SESSION ID (UID)
        self.UID = datetime.now().strftime('%m%d%H%M')

    def get_sessid(self):
        return self.UID

    def get_config(self):
        return self.CUR_MODEL, self.BATCH_SIZE, self.NUM_WORKERS, self.EPOCHS

    def get_consts(self):
        return self.CLASS_MAPPING, self.MODELS, self.NUM_LABELS