import numpy as np
from scipy.interpolate import interp1d

class BatteryDataProcessor:
    def __init__(self):
        self.v_min, self.v_max = 2.0,3.6
        self.t_min, self.t_max = 26.0,42.0
        self.target_length = 1024

    def interpolate_and_scale(self, time_raw, voltage_raw, temp_raw):
        time_uniform = np.linspace(time_raw[0],time_raw[-1],self.target_length)
        f_voltage=interp1d(time_raw,voltage_raw,kind='linear',fill_value="extrapolate")
        f_temp=interp1d(time_raw,temp_raw,kind='linear',fill_value="extrapolate")
        v_interp=f_voltage(time_uniform)
        t_interp=f_temp(time_uniform)
        v_scaled=(v_interp-self.v_min)/(self.v_max-self.v_min)
        t_scaled=(t_interp-self.t_min)/(self.t_max- self.t_min)
        v_scaled=np.clip(v_scaled,0.0,1.0)
        t_scaled=np.clip(t_scaled,0.0,1.0)
        return v_scaled, t_scaled

    def build_feature_vector(self, v_scaled, t_scaled, discharge_time,cycle_norm):
        scalars = np.array([discharge_time, cycle_norm])
        feature_vector =np.concatenate([v_scaled,t_scaled,scalars])
        return feature_vector
