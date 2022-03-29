import enum


#TODO naming..
class DeploymentType(enum.Enum):
    LOCAL = 1
    #expose GPUs
    LOCAL_AML = 2
    AML_ON_AKS = 3


MII_CACHE_PATH = "MII_CACHE_PATH"
MII_CACHE_PATH_DEFAULT = "/tmp/mii_cache"
