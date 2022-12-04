"""Unit conversions for variables used in merra2 and era5 data."""
import numpy as np

CLAVRX_FILL = 9.999e20
COMPRESSION_LEVEL = 6


def meter_to_km(data):
    """Convert height from m to km."""
    return data / 1000.0


def pa_to_hPa(data):
    """Convert from Pa to hPa."""
    return data / 100.0


def no_conversion(data):
    """Return the data without conversion."""
    return data


def scale_tpw(data):
    """Return scale mm to cm."""
    return data / 10.0


def fill_bad(data):
    """Fill with np.nan."""
    return data * np.nan


def geopotential(data):
    """Convert geopotential in meters per second squared to geopotential height."""
    # this is height/1000.0*g
    return data / 9806.65


def kg_per_metersq_to_dobson(data):
    """Convert kg/m^2 to dobson units."""
    return data / 2.1415e-5


def pressure_to_altitude(pressure):
    """Use surface pressure to converted to km rather than surface geopotential for altitude."""
    P_zero = 101325  # Pa (Pressure at altitude 0)
    T_zero = 288.15  # K (Temperature at altitude 0)
    g = 9.80665  # m/s^2 (gravitational acceleration)t
    L = -6.50E-03  # K/m (Lapse Rate)
    R = 287.053  # J/(KgK) (Gas constant for air)

    altitude = (T_zero / L) * ((pressure / P_zero) * np.exp(-L * R / g) - 1)

    altitude = altitude.astype(np.float32)

    return altitude


def km_per_square_meter_to_cm(data):
    """Convert kg/m2 of water to cm."""
    return data*0.1
