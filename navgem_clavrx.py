#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  This code produces CLAVRx compatible files from navgem global model data.
#
#  filter_by_keys is used to select data from the grib into xarray open_dataset
#  however, it is easier to see the data using cfgrib directly.  cfgrib will
#  create a list of loaded xarray Datasets:
#  `import cfgrib
#   cfgrib.open_datasets(<grib_file>)`
#
"""Convert navgem model data into CLAVRx compatible input."""
from __future__ import annotations

import glob
import json
import itertools
import logging
import os
import parse
import shutil
import tempfile

import yaml

import navgem_retrieve as navgem_get
#from testing import printValue

try:
    import argparse
    import datetime
    import dateutil
    from typing import Callable, Dict, List, Optional, TypedDict
    from datetime import datetime as dt

    import numpy as np
    import pandas as pd
    import xarray as xr
    from pyhdf.SD import SD, SDC
except ImportError as e:
    msg = "{}.  Try 'conda activate merra2_clavrx'".format(e)
    raise ImportError(msg)

from conversions import CLAVRX_FILL, COMPRESSION_LEVEL

LOG = logging.getLogger(__name__)

OUT_PATH_PARENT = '/apollo/cloud/Ancil_Data/clavrx_ancil_data/dynamic/navgem/'

class NavgemCommandLineMapping(TypedDict):
    """Type hints for result of the argparse parsing."""

    start_date: datetime.date
    end_date: Optional[str]
    base_path: str
    input_path: str
    forecast_hours: List[int]
    local: bool
    model_run: str


class DateParser(argparse.Action):
    """Parse a date from argparse to a datetime."""

    def __call__(self, parser, namespace, values, option_strings=None):
        """Parse a date from argparse to a datetime."""
        setattr(namespace, self.dest, dateutil.parser.parse(values).date())


def output_dtype(out_name, nc4_dtype):
    """Convert between string and the equivalent SD.<DTYPE>."""
    if (nc4_dtype == "single") | (nc4_dtype == "float32"):
        sd_dtype = SDC.FLOAT32
    elif (nc4_dtype == "double") | (nc4_dtype == "float64"):
        sd_dtype = SDC.FLOAT64
    elif nc4_dtype == "uint32":
        sd_dtype = SDC.UINT32
    elif nc4_dtype == "int32":
        sd_dtype = SDC.INT32
    elif nc4_dtype == "uint16":
        sd_dtype = SDC.UINT16
    elif nc4_dtype == "int16":
        sd_dtype = SDC.INT16
    elif nc4_dtype == "int8":
        sd_dtype = SDC.INT8
    elif nc4_dtype == "char":
        sd_dtype = SDC.CHAR
    else:
        raise ValueError("UNSUPPORTED NC4 DTYPE FOUND:", nc4_dtype)

    if out_name in ["pressure levels", "level"] and sd_dtype == SDC.FLOAT64:
        sd_dtype = SDC.FLOAT32  # don't want double

    return sd_dtype


def read_yaml():
    """Read the yaml file with setup variables."""
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(ROOT_DIR, 'yamls', 'NAVGEM_nrl_usgodae_vars.yaml'), "r") as yml:
        OUTPUT_VARS_DICT = yaml.load(yml, Loader=yaml.Loader)

    return OUTPUT_VARS_DICT


def set_dim_names(data_array, ndims_out, out_name, out_sds):
    """Set dimension names in hdf file."""
    if ndims_out in (2, 3):
        coords_dict = {"z": (2, "level"),
                       "x": (1, "lon"),
                       "y": (0, "lat"),
                       }
    else:
        coords_dict = {"z": (0, "level"),
                       "x": (0, "lon"),
                       "y": (0, "lat"),
                       }

    out_sds.dimensions()
    msg_str = "Out {} for {} ==> {}."
    msg_str = msg_str.format(out_sds.dimensions(),
                             data_array.name,
                             out_name)
    LOG.debug(msg_str)
    for dim in data_array.dims:
        axis_num, dim_name = coords_dict.get(dim, dim)

        out_sds.dim(axis_num).setname(dim_name)

    msg_str = "Becomes {} for {} ==> {}."
    msg_str = msg_str.format(out_sds.dimensions(),
                             data_array.name,
                             out_name)
    LOG.debug(msg_str)

    return out_sds


def refill(data: xr.DataArray, old_fill: float) -> np.ndarray:
    """Assumes CLAVRx fill value instead of variable attribute."""
    data = data.fillna(CLAVRX_FILL)
    if data.dtype in (np.float32, np.float64):
        if old_fill != CLAVRX_FILL:
            data = xr.where(data == old_fill, CLAVRX_FILL, data)
    return data


def update_output(sd, out_name, rsk, data_array, out_fill, data_source):
    """Finalize output variables."""
    out_units = rsk["out_units"]
    ndims_out = rsk["ndims_out"]
    str_template = f"Writing Input name: {data_array.long_name} ==> Output Name: {out_name}"
    LOG.info(str_template)
    data_array = reshape(data_array, out_name, ndims_out)

    out_sds = sd["out"].create(out_name,
                               output_dtype(out_name, data_array.dtype),
                               data_array.shape)
    out_sds.setcompress(SDC.COMP_DEFLATE, value=COMPRESSION_LEVEL)
    set_dim_names(data_array, ndims_out, out_name, out_sds)
    if out_name == "lon":
        out_sds.set(data_array.data)
    else:
        out_data = refill(data_array, out_fill)
        out_sds.set(out_data)

    if out_fill is not None:
        out_sds.setfillvalue(CLAVRX_FILL)

    if out_units is None or out_units in ("none", "None"):
        try:
            out_sds.units = data_array.units
        except AttributeError:
            out_sds.units = "1"
    else:
        out_sds.units = out_units

    unit_desc = " in [{}]".format(out_sds.units)

    out_sds.source_data = ("{}->{}{}".format("{}->{}".format(data_source, data_array.name),
                                             data_array.name, unit_desc))

    out_sds.long_name = data_array.long_name
    out_sds.endaccess()


def modify_shape(data_array: xr.DataArray) -> xr.DataArray:
    """Modify shape from dims (level, lat, lon) to (lat,lon,level)."""
    ndim = len(data_array.dims)
    if ndim == 3:
        return data_array.transpose("y", "x", "z")
    if ndim == 2:
        return data_array.transpose("y", "x")
    else:
        return data_array


def reshape(data_array: xr.DataArray,
            out_name: str, ndims_out: int) -> xr.DataArray:
    """Reshape data toa->surface and (lat, lon, level)."""
    if "isobaricInhPa" in data_array.dims:
        # clavr-x needs toa->surface not surface->toa
        data_array = data_array.sortby(data_array["isobaricInhPa"], ascending=True)

    return modify_shape(data_array)


def all_equal(iterable):
    """Return True if all the elements are equal to each other."""
    g = itertools.groupby(iterable)
    return next(g, True) and not next(g, False)


def starmap(function, iterable):
    """Itertools apply function to iterable."""
    # starmap(pow, [(2,5), (3,2), (10,3)]) --> 32 9 1000
    for args in iterable:
        yield function(*args)


def obtain_gfs_fn(start_time: datetime.datetime, model_init_hour: str,
                  forecast_hour: str, gfs_load_crc) -> str:
    """Use information from ozone mixing ratio yaml setup to build filepath.

    :param start_time: start_time is part of the filename format string and must match yaml
        (so if this variable changes here, it should change in the yaml as well.)
    :param model_init_hour: model_init_hour is part of a filename format string and must match yaml
        (so if this variable changes here, it should change in the yaml as well.)
    :param forecast_hour: forecast_hour is part of a filename format string and must match yaml
        (so if this variable changes here, it should change in the yaml as well.)
    :param gfs_load_crc: This is the datasets section of the yaml and will contain
         the appropriate filepath and filename pattern for the source files.
    :return: The full filepath based on these input of the ozone mixing ratio supplement file.
    """
    
    gfs_dir = gfs_load_crc.pop("directory")
    gfs_pattern = gfs_load_crc.pop("file_pattern")
    # both had an eval next two lines
    base_dir = eval(gfs_dir)
    base_fn = eval(gfs_pattern)

    # read complementary GFS 03MR here
    full_filepath = os.path.join(base_dir, base_fn)

    imsg = f"{start_time}, {model_init_hour}, F{forecast_hour} => {full_filepath}"
    LOG.info(imsg)

    return full_filepath, gfs_load_crc


def read_gfs(gfs_dict, model_init, model_run_hour, forecast) -> xr.Dataset:
    """Use the GFS ozone mixing ratio.

    :param o3mr_fn: filepath of the gfs file.
    :return: ozone mixing ratio in kg/kg
    """

    gfs_fn, ds_var_dict = obtain_gfs_fn(model_init, model_run_hour,
                                        str(forecast), gfs_dict) 
     

    dirpath = tempfile.mkdtemp(dir=os.path.expanduser("~"))

    gfs_datasets_arr = []
    for da_key, new_filters in ds_var_dict.items():
        gfs_datasets_arr.append(load_dataset(gfs_fn, model_run_hour, 'gfs_grib', new_filters))
                                              
    gfs_ds = xr.merge(gfs_datasets_arr)
    gfs_ds = gfs_ds.transpose("latitude", "longitude", "isobaricInhPa")
    gfs_ds = gfs_ds.sortby("latitude", ascending=True)

    shutil.rmtree(dirpath)

    return gfs_ds


def reformat_levels(in_ds, key_name):
    """Verify output levels of dataset are on CLAVRx levels."""
    hPa_levels = [1000.0, 975.0, 950.0, 925.0, 900.0, 850.0,
                  800.0, 750.0, 700.0, 650.0, 600.0, 550.0,
                  500.0, 450.0, 400.0, 350.0, 300.0, 250.0,
                  200.0, 150.0, 100.0, 70.0, 50.0, 30.0, 20.0, 10.0]

    try:
        in_ds = in_ds.sel(isobaricInhPa=hPa_levels)
    except KeyError as kerr:
        ke_msg = "{} for {} in coords".format(kerr, key_name)
        LOG.warning(ke_msg)
    return in_ds


def apply_conversion(scale_func: Callable, data: xr.DataArray, fill) -> xr.DataArray:
    """Apply fill to converted data after function."""
    converted = data.load().copy()
    converted = scale_func(converted)

    if data.dims == converted.dims:
        if fill is not None:
            converted = xr.where(data == fill, fill, converted)
        if np.isnan(data).any():
            converted = xr.where(np.isnan(data), fill, converted)

    converted = converted.astype(np.float32)
    converted = converted.assign_attrs(data.attrs)

    return converted


def write_output_variables(in_datasets, out_vars_setup: Dict):
    """Write variables to file."""
    for var_key, rsk in out_vars_setup.items():
        #if var_key in ["rh", "rh_level"]:
        #    file_key = "rltv_hum"
        #else:
        file_key = rsk["dataset"]
        var_name = rsk["cfVarName"]
        out_var = in_datasets[file_key][var_name]
        units_fn = rsk["units_fn"]
        if "data_source_format" in rsk.keys():
            source_model = rsk["data_source_format"]
        else:
            source_model = "NAVGEM"

        try:
            var_fill = out_var.fill_value
        except AttributeError:
            var_fill = 1e+20  # match merra2 :/

        # return a new xarray with converted data, otherwise, the process
        # is different for coordinate attributes.
        out_var = apply_conversion(units_fn, out_var, fill=var_fill)

        update_output(in_datasets, var_key, rsk,
                      out_var, var_fill, source_model)


def get_dim_list_string(param: Dict[str]) -> str:
    """Create an attribute string from the dims."""
    dim_list = []
    for dim_name in param.keys():
        if dim_name.lower() in ["lat", "latitude"]:
            out_dim = 'Y'
        elif dim_name.lower() in ["lon", "longitude"]:
            out_dim = 'X'
        elif dim_name.lower() in ["level", "pressure", "rh_level",
                                  "press", "gph", "height"]:
            out_dim = 'Z'
        else:
            out_dim = dim_name
        dim_list.append(out_dim)
    return "".join(dim_list)


def write_global_attributes(out_ds: SD, model_run, valid_time, tpw_time, cape_time, forecast) -> None:
    """Write global attributes."""

    var = out_ds.select('temperature')
    nlevel = var.dimensions(full=False)['level']
    nlat = var.dimensions(full=False)['lat']
    nlon = var.dimensions(full=False)['lon']
    setattr(out_ds, 'NUMBER OF LATITUDES', nlat)
    setattr(out_ds, 'NUMBER OF LONGITUDES', nlon)
    setattr(out_ds, 'NUMBER OF PRESSURE LEVELS', nlevel)
    setattr(out_ds, 'NUMBER OF O3MR LEVELS', nlevel)
    setattr(out_ds, 'NUMBER OF CLWMR LEVELS', nlevel)
    setattr(out_ds, 'MODEL INITIALIZATION TIME', model_run.strftime("%Y-%m-%d %HZ"))
    setattr(out_ds, 'FORECAST', forecast)
    setattr(out_ds, 'CAPE FORECAST', cape_time)
    setattr(out_ds, 'TPW FORECAST', tpw_time)
    setattr(out_ds, 'VALID TIME', valid_time.strftime("%Y-%m-%d %HZ"))
    setattr(out_ds, 'VALID TIME', valid_time.strftime("%Y-%m-%d %HZ"))
    lat = out_ds.select('lat')
    lon = out_ds.select('lon')
    attr = out_ds.attr('LATITUDE RESOLUTION')
    attr.set(SDC.FLOAT32, (lat.get()[1] - lat.get()[0]).item())
    attr = out_ds.attr('LONGITUDE RESOLUTION')
    attr.set(SDC.FLOAT32, (lon.get()[1] - lon.get()[0]).item())
    attr = out_ds.attr('FIRST LATITUDE')
    attr.set(SDC.FLOAT32, (lat.get()[0]).item())
    attr = out_ds.attr('FIRST LONGITUDE')
    attr.set(SDC.FLOAT32, (lon.get()[0]).item())

    dim_description = get_dim_list_string(var.dimensions())
    setattr(out_ds, '3D ARRAY ORDER', dim_description)  # XXX is this true here?
    [a.endaccess() for a in [var, lat, lon]]

    out_ds.end()


def reorder_dimensions(ds):
    """Reorder and rename dimensions."""
        # rename_dims
    for dim_key in ds.dims:
        dim = ds[dim_key]
        if dim.long_name.lower() == "longitude":
            ds = ds.rename_dims({dim.name: "x"})
        elif dim.long_name.lower() == "latitude":
            ds = ds.rename_dims({dim.name: "y"})
        elif dim.long_name.lower() == "pressure":
            ds = ds.rename_dims({dim.name: "z"})

    return ds


def load_dataset(model_file: str, model_run_hour, dataset_key, filters):
    """Use cfgrib to load NAVGEM model data."""

    # make a temp directory for the cfgrib idx file
    LOG.debug("Read {}".format(model_file))
    LOG.info(filters)

    # in general, need to select 12 hour forecast, except
    # in cases when PWAT is being pulled from a previous
    # model run
    if dataset_key in ["gfs_grib"]:
        pass
    else:
        # select 12 hour forecast for all data every model run (can only get up to 12Z forecast from cape)
        # For TPW, select the 18 Z forecast for the 6, 18 runs since tpw is only generated at 0 and 12 model runs
        filters.update({"P1": 12})
        if model_run_hour in ["06", "18"] and dataset_key == "prcp_h20":
            filters.update({"P1": 18})  # from the 00Z and 12Z model runs respectively.

    dirpath = tempfile.mkdtemp(dir=os.path.expanduser("~"))
    cmd = " xr.open_dataset('{}', engine='cfgrib', backend_kwargs={{'filter_by_keys': {}}})"
    cmd = cmd.format(model_file, json.dumps(filters))
    LOG.debug(cmd)
    ds = xr.open_dataset(model_file, engine="cfgrib",
                         backend_kwargs={'filter_by_keys': filters, 
                                         "indexpath": dirpath + "/input.{short_hash}.idx"})

    shutil.rmtree(dirpath)

    # check if empty
    if len(ds.sizes) < 1:
        err_msg = "{} empty with backend_kwargs={}'filter_by_keys': {}{}"
        err_msg = err_msg.format(model_file, "{", filters, "}")
        raise ValueError(err_msg)
    return ds


def read_one_hour_navgem(file_dict: dict,
                         out_dir: str,
                         model_initialization: datetime,
                         forecast_hour: int):
    """Read input, parse times, and run conversion on one day at a time."""
    datasets = dict()
    timestamps = list()
    params = list()

    # build time selection based on forecast hour and model date.
    valid_time = model_initialization + pd.Timedelta("{} hours".format(forecast_hour))
    model_hour = model_initialization.strftime("%H")

    fh = str(forecast_hour).zfill(3)
    navgem_fn_pattern = f"navgem.%y%m%d{model_hour}_F{fh}.hdf"
    out_fname = model_initialization.strftime(navgem_fn_pattern)
    tpw_str = "" 
    cape_str = "" 

    dvar_yaml = read_yaml()

    for ds_key, ds_dict in dvar_yaml["datasets"].items():
        if ds_key == "gfs_grib":
            ds = read_gfs(ds_dict, model_initialization, model_hour, fh)
        else:
            try:
                fe = ds_dict["file_ending"]
            except:
                fe = ds_key

            model_file = file_dict[fe]

            ds = load_dataset(model_file, model_hour, ds_key, ds_dict["filters"])

        if ds_key in ["prcp_h20", "cape"]:
            tpw_ts = pd.to_datetime(ds.time.data)
            delta = ds.step.data
            print(ds_key, tpw_ts, delta)
            step = int(delta.astype("timedelta64[h]") / np.timedelta64(1, 'h') % 24)
            fc_str = "{} {}Z forecast".format(tpw_ts.strftime("%Y-%m-%d %HZ"), step)
            if ds_key == "prcp_h20":
                tpw_str = fc_str
            else:
                cape_str = fc_str 
        else:
            ts = pd.to_datetime(ds.valid_time.data)
            params.append(ds_key)
            timestamps.append(ts)
            delta = ds.step.data
            step = int(delta.astype("timedelta64[h]") / np.timedelta64(1, 'h') % 24)
            forecast_str= "{}Z".format(step)


        ds = reformat_levels(ds, ds_key)
        ds = reorder_dimensions(ds)
        datasets.update({ds_key: ds})

    if all_equal(timestamps):
        pass
    else:
        ts_msg = "Timestamps are not equal"
        for var, ts in zip(params, timestamps):
            print(var,ts)
        raise ValueError(ts_msg)
        
    if timestamps[0] == valid_time:
        out_fname = os.path.join(out_dir, out_fname)
        LOG.info('    working on {}'.format(out_fname))

        # TRUNC will clobber existing
        datasets['out'] = SD(out_fname, SDC.WRITE | SDC.CREATE | SDC.TRUNC)

        write_output_variables(datasets, dvar_yaml["data_arrays"])
    else:
        ts_str = timestamps[0].strftime("%Y-%m-%d %H")
        ts_msg = "Timestamps are not equal/not equal to valid_time: {} {}".format(ts_str,
                                                                             valid_time)
        raise ValueError(ts_msg)

    write_global_attributes(datasets['out'], model_initialization,
                            valid_time, tpw_str, cape_str, forecast_str)

    return out_fname


def get_model_run_string(model_date_dt, run_hour):
    """Given a model date and model run hour, create a model run string."""
    model_date = model_date_dt.strftime("%Y%m%d")

    md_msg = "Model Date {}: {}".format(type(model_date), model_date)
    LOG.debug(md_msg)
    md_msg = "Run Hour (model run) {}: {}".format(type(run_hour), run_hour)
    LOG.debug(md_msg)

    dt_model_run = dt.strptime("{} {}".format(model_date, run_hour), "%Y%m%d %H")
    model_run_str = dt_model_run.strftime("%Y%m%d%H")

    return model_run_str


def build_filepath(data_dir, dt: datetime, dir_type="output") -> str:
    """Create output path from in put model run information."""
    year = dt.strftime("%Y")
    year_month_day = dt.strftime("%Y_%m_%d")
    if dir_type == "output":
        this_filepath = os.path.join(data_dir, year, year_month_day)
        LOG.info(f"Making {this_filepath}")
        os.makedirs(this_filepath, exist_ok=True)
    elif dir_type == "input":
        this_filepath = os.path.join(data_dir, year, year_month_day, "nrl_orig")
        os.makedirs(this_filepath, exist_ok=True)
    else:
        raise RuntimeError('dir_type options are either ["input", "output"]')

    return this_filepath


def process_navgem(base_path=None, input_path=None, start_date=None,
                   url=None, model_run=None, forecast_hours=None,
                   local=False) -> None:
    """Read input, parse times, and run conversion."""
    if local:
        raise RuntimeError("Local is true, but process_navgem subroutine pulls data.")

    out_list = None

    grib_path = build_filepath(input_path, start_date, dir_type="input")
    os.makedirs(grib_path, exist_ok=True)

    # start_time is needed for url string.
    start_time = dt.combine(start_date, datetime.time(int(model_run)))
    LOG.debug("model run {}".format(start_time.strftime("%Y%m%d%H")))
    url = eval(url)
    soup = navgem_get.create_soup(url)

    out_fp = build_filepath(base_path, start_date)
    model_run_str = get_model_run_string(start_date, model_run)

    model_run_dt = dt.strptime(model_run_str, "%Y%m%d%H")

    # The data download may need more than the actual forecast run
    in_files = navgem_get.url_search_nrl(soup, url, model_run_dt, dest_path=grib_path)

    for forecast_hour in forecast_hours:
        out_list = read_one_hour_navgem(in_files, out_fp, start_time, forecast_hour)
    LOG.info(out_list)


def argument_parser() -> NavgemCommandLineMapping:
    """Parse command line for navgem_clavrx.py."""
    parse_desc = (
        """\nProcess navgem data previously downloaded from NCEP nomads or NRL ftp.""")

    setup_yaml = read_yaml()
    url = setup_yaml["url"]

    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=parse_desc,
                                     formatter_class=formatter)
    group = parser.add_mutually_exclusive_group()

    parser.add_argument('start_date', action=DateParser,
                        default=dt.now(), help="Processing date")
    parser.add_argument('-m', '--model_run', default='00',
                        help="Model run hour; i.e. 00, 03, 06, 09, 12...")
    parser.add_argument('-f', '--forecast_hours', nargs='+',
                        default=[3, 6, 9, 12, 18],
                        help="The forecast hours from this model run.")

    group.add_argument('-u', '--url', default=url,
                       help='alternative url string.')
    parser.add_argument('-i', '--input', dest='input_path', action='store',
                        type=str, required=False, default=None, const=None,
                        help="Input path for the data download.")
    # store_true evaluates to False when flag is not in use (flag invokes the store_true action)
    group.add_argument('-l', '--local', action='store_true',
                       help="Use local files already in input path.")
    parser.add_argument('-d', '--base_path', action='store', nargs='?',
                        type=str, required=False, default=OUT_PATH_PARENT, const=OUT_PATH_PARENT,
                        help="Parent output path: year subdirectory appends to this path.")
    parser.add_argument('-v', '--verbose', dest='verbosity', action="count", default=2,
                        help='each occurrence increases verbosity 1 level through '
                             'ERROR-WARNING-INFO-DEBUG')

    args = vars(parser.parse_args())
    verbosity = args.pop('verbosity', None)

    levels = [logging.ERROR, logging.WARN, logging.INFO, logging.DEBUG]
    logging.basicConfig(format='%(module)s:%(funcName)s:%(lineno)d:%(levelname)s:%(message)s',
                        level=levels[min(3, verbosity)])

    return args


if __name__ == '__main__':

    parser_args = argument_parser()
    fn_dict = dict()

    if parser_args["local"]:
        inp = parser_args["input_path"]
        inp = build_filepath(inp, parser_args["start_date"], dir_type="input")

        dt_in = parser_args["start_date"]
        model_str = get_model_run_string(dt_in, parser_args["model_run"])
        model_dt = dt.strptime(model_str, "%Y%m%d%H")

        # get every file in directory
        full_glob = os.path.join(inp, "navgem_{}_*.grib".format(model_str))
        LOG.debug("Process glob: {}".format(full_glob))
        fn_paths = glob.glob(full_glob)

        if len(fn_paths) > 0:
            LOG.debug("Found: {}".format(fn_paths))
        else:
            raise RuntimeError("No files found using {}".format(full_glob))

        # find keys for this model run
        pattern = "{inpath}navgem_{modelrun}_{file_ending}.grib"
        results = list(parse.parse(pattern, x) for x in fn_paths)

        for line in results:
            inp = (line["inpath"])
            run = (line["modelrun"])
            param  = (line["file_ending"])
            # get run_adjustment
            run_dt = navgem_get.model_run_adjustment(dt.strptime(run, "%Y%m%d%H"), param)
            run = run_dt.strftime("%Y%m%d%H")
            param_grib = f"{inp}/navgem_{run}_{param}.grib"
            try:
                fn_dict.update({param: param_grib})
            except KeyError as key_err:
                raise 

        out_fnames = []
        out_path = build_filepath(parser_args['base_path'], dt_in)

        for forecast in parser_args["forecast_hours"]:
            out_fnames.append(read_one_hour_navgem(fn_dict, out_path,
                                                   model_dt, forecast))
            LOG.info(out_fnames)
    else:
        process_navgem(**parser_args)
