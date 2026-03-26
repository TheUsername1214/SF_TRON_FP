from SF_TRON_FP.SRC.Config.Config import *

PPOCfg.CriticParam.state_dim = 33 + 18 * 11 + PPOCfg.EstimatorParam.output_dim
