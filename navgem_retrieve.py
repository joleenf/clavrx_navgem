"""Using Beautiful Soup, download data for navgem."""
import logging
import os
import re
import shlex
import warnings
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from datetime import datetime as datetime, timedelta
from glob import glob as glob
from subprocess import PIPE, Popen

import requests

OUT_PATH_PARENT = "/Users/joleenf/data/clavrx/navgem/nrl/"

try:
    from bs4 import BeautifulSoup
except ImportError as ie:
    raise ImportError("{}\n Please activate environment: conda activate merra2_clavrx".format(ie))

LOG = logging.getLogger(__name__)


def download_file(url, filename, destination="data/joleenf/navgem/data/"):
    """Download filename from url to destination."""
    LOG.info("Download {}".format(url))
    response = requests.get(url, verify=False, stream=True)
    try:
        response.raise_for_status()
    except Exception as err:
        msg = "Could not retrieve data {}".format(err)
        warnings.warn(msg)
        return None

    data_dir = "{}/{}".format(destination, filename)
    LOG.debug("Download {}".format(data_dir))
    open(data_dir, "wb").write(response.content)

    return data_dir


def download_url(url):
    """Download file from url given."""
    print("downloading: ", url)
    # assumes that the last segment after the / represents the file name
    # if url is abc/xyz/file.txt, the file name will be file.txt
    file_name_start_pos = url.rfind("/") + 1
    file_name = url[file_name_start_pos:]

    r = requests.get(url, stream=True)
    if r.status_code == requests.codes.ok:
        with open(file_name, 'wb') as f:
            for data in r:
                f.write(data)


def create_soup(URL):
    """Create soup object from page request."""
    page = requests.get(URL, verify=False)
    page.raise_for_status()
    LOG.debug(f"Send {URL} to Soup Parser")
    soup = BeautifulSoup(page.content, "html.parser")

    return soup


def url_search_by_filenames(url_soup, url, get_these_files, out_path):
    """Provide a list of filenames and search within given url for download."""
    list_of_files = []
    for a in url_soup.find_all('a', href=True):
        if a['href'] == a.text and a.text in get_these_files:
            url_fn = url + "/" + a.text
            dl_file = download_file(url_fn, a.text, destination=out_path)
            if isinstance(dl_file, str):
                list_of_files.append(dl_file)
    if list_of_files:
        return list_of_files
    else:
        raise RuntimeError(f"No NAVGEM files loaded with {url}")


def build_curl_file_list(bs4_soup, url, regex_pattern, destination, download=True):
    """Create the curl file list matching regex."""
    # need to get precipitable water which is not in every model run?
    list_of_files = []
    for link in bs4_soup.find_all('a'):
        print("Found: {}".format(link.text))
        a=(re.search(regex_pattern, link.text))
        if a is not None:
            url_fn = url + "/" + link.text
            list_of_files.append(url_fn)

    if list_of_files and download:
        LOG.debug(len(list_of_files))
        file_list = ' '.join(list_of_files)

        args = shlex.split("curl --output-dir {} --progress-bar"
                           "--insecure --remote-name-all {}".format(destination, file_list))

        ip = Popen(args, stdin=PIPE, stdout=PIPE)
        LOG.debug(ip.communicate())

    else:
        LOG.info(regex_pattern + ": " + " ".join(list_of_files))

    return list_of_files


def model_run_adjustment(run_dt, field_name):
    """Adjust the model run time back to 00Z or 12Z for some fields not produced at 06Z and 18Z."""

    adjust_fields = ["prcp_h20", "cape"]

    hour_of_run = run_dt.strftime("%H")
    # date transitions at 18Z run + 12 hours.
    if hour_of_run in ["06", "18"] and field_name in adjust_fields:
        run_dt = run_dt - timedelta(hours=6)
    return run_dt


def get_product_set_nrl(this_url, this_soup, this_dest_path, this_run_str,
                        this_products_list, download=True):
    """Use this set of urls and products to get data."""

    file_dict = dict()
    pattern = list()
    for product_name in this_products_list:
        pattern.append("US.*?{}.*?{}_(\d+)_(\d+)-(\d+){}{}".format(r"(\d+)", this_run_str, product_name, r"\b"))

        LOG.debug("{}: {}".format(product_name, pattern))
        LOG.debug("{}".format(this_url))
    all_patterns = "|".join(pattern)
    LOG.debug(all_patterns)

    product_files = build_curl_file_list(this_soup, this_url,  all_patterns, this_dest_path, download=download)
    LOG.debug(product_files)

    if download:
        for product_name in this_products_list:
            grib_name = concat_gribs_to_many(this_dest_path, dt(this_run_str, "%Y%m%d%H"), product_name)
            file_dict.update({product_name: grib_name})
    return file_dict


def url_search_nrl(url_soup, url, navgem_run_dt, forecast=[12], dest_path=None, products_list=None):
    """Get NRL data using regex."""
    # Product name model table:  OLD INFO: https://www.usgodae.org/docs/layout/pn_model_tbl.pns.html
    if products_list is None:
        products_list = ["pres_msl", "pres", "rltv_hum", "air_temp", "snw_dpth",
                         "wnd_ucmp", "wnd_vcmp", "geop_ht", "terr_ht", "air_temp",
                         "vpr_pres", "ice_cvrg", "cape", "prcp_h20"]


    navgem_run = navgem_run_dt.strftime("%Y%m%d%H")
    # dataset ID table:  https://www.usgodae.org/docs/layout/pn_dataset_tbl.pns.html

    full_list = dict() 
    tpw_soup = None
    url_tpw = None
    special_list = []

    if "prcp_h20" in products_list or "cape" in products_list:
        # date transitions at 18Z run + 12 hours.
        tpw_run = model_run_adjustment(navgem_run_dt, "cape")
        tpw_run_str = tpw_run.strftime("%Y%m%d%H")
        year_str = tpw_run.strftime("%Y")
    
        url_base = os.path.dirname(os.path.dirname(url))
        url_tpw = os.path.join(url_base, year_str, tpw_run_str)

        if url != url_tpw and tpw_soup is None:
            tpw_soup = create_soup(url_tpw)
            # drop from product_list
            special_list = [e for e in products_list if e in ["prcp_h20", "cape"]]
            products_list = [e for e in products_list if e not in ["prcp_h20", "cape"]]

    full_list.update(get_product_set_nrl(url, url_soup, dest_path, navgem_run, products_list))

    if tpw_soup is None:
        full_list.update(get_product_set_nrl(url_tpw, tpw_soup, dest_path, tpw_run_str, special_list))

    if len(full_list) == 0:
        runtime_msg = f"No NAVGEM files loaded with {url}"
        raise RuntimeError(runtime_msg)
    return full_list


def search_date(url_soup, url, navgem_run_dt, forecast_times, output_path="."):
    """From soup, get file list matching date and forecast times."""
    navgem_run = navgem_run_dt.strftime("%Y%m%d%H")
    if forecast_times is None:
        forecast_times = [0, 6, 12, 18]
    get_these_files = []
    for forecast in forecast_times:
        forecast = str(forecast).zfill(2)
        # TODO:  Check if this is correct (This is NOMADS format)
        file_name = f'navgem_{navgem_run}f{forecast}.grib2'
        get_these_files.append(file_name)

    LOG.info(get_these_files)
    downloaded_files = url_search_by_filenames(url_soup, url, get_these_files, output_path)

    return downloaded_files


def concat_gribs_to_many(data_path, model_run_dt, file_ending):
    """Concat each variable into it's own grib file.

       In some cases, this will just be a cat of an original
       single file to a new input name.  This is silly work, but
       keeps the process consistent.
    """

    found_files = glob(os.path.join(data_path, "*{}".format(file_ending)))

    model_run_dt = model_run_adjustment(model_run_dt, file_ending)
    model_run = model_run_dt.strftime("%Y%m%d%H")
    if len(found_files) > 0:
        model_run_glob = f"US*{model_run}*{file_ending}"
        data_path_glob = os.path.join(data_path, model_run_glob)

        grib_name = os.path.join(data_path, f'navgem_{model_run}_{file_ending}.grib')
        os.system(f'cat {data_path_glob} > {grib_name}')
    else:
        raise FileNotFoundError("MISSING NRL {} product files for {}".format(file_ending, model_run))

    return grib_name


def concat_gribs_in_one(data_path, model_run):
    """Concat all files in data_path, using model_run to name output file."""
    grib_name = os.path.join(data_path, f'navgem_{model_run}.grib')
    model_run_glob = f"US*{model_run}*"
    data_path_glob = os.path.join(data_path, model_run_glob)
    os.system(f'cat {data_path_glob} > {grib_name}')

    # Need to cat the TPW file into this grib set
    model_run_hour = model_run[-2:]
    model_run_date = model_run[:-2]
    if model_run_hour == "06":
        model_run_tpw = model_run_date + "00"
    elif model_run_hour == "18":
        model_run_tpw = model_run_date + "12"
    else:
        return grib_name

    tpw_run_glob = f"US*{model_run_tpw}*prcp_h20"
    data_path_glob_tpw = os.path.join(data_path, tpw_run_glob)
    os.system(f'cat {data_path_glob_tpw} >> {grib_name}')

    return grib_name


def argument_parser():
    """Parse command line for navgem_clavrx.py."""
    parse_desc = (
        """\nGet navgem data from downloaded from USGODAE server.""")

    formatter = ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=parse_desc,
                            formatter_class=formatter)

    parser.add_argument('-s', '--start_date', type=str,
                        default=datetime.now().strftime('%Y%m%d'),
                        help="Desired processing date as YYYYMMDD")
    parser.add_argument('-r', '--run_hour', action='store',
                        type=str, required=False, default='00',
                        help="Two digit model run hour.")
    parser.add_argument('-f', '--forecast_hours', nargs='+',
                        default=[3, 6, 12, 18],
                        help="The forecast hours.")
    parser.add_argument('-d', '--base_path', action='store', nargs='?',
                        type=str, required=False, default=OUT_PATH_PARENT, const=OUT_PATH_PARENT,
                        help="Parent path: year subdirectory appends to this path.")
    parser.add_argument('-v', '--verbose', dest='verbosity', action="count", default=2,
                        help='each occurrence increases verbosity 1 level through '
                             'ERROR-WARNING-INFO-DEBUG')

    args = vars(parser.parse_args())
    verbosity = args.pop('verbosity', None)

    levels = [logging.ERROR, logging.WARN, logging.INFO, logging.DEBUG]
    logging.basicConfig(format='%(module)s:%(lineno)d:%(levelname)s:%(message)s',
                        level=levels[min(3, verbosity)])

    year = args['start_date'][:4]
    run_hour = args['run_hour'].zfill(2)

    model_run = f"{args['start_date']}{run_hour}"

    #args['base_path'] = os.path.join(args['base_path'], year, model_run)

    return args


if __name__ == '__main__':

    parser_args = argument_parser()

    model_run = f"{parser_args['start_date']}{parser_args['run_hour']}"
    model_run = datetime.strptime(model_run, "%Y%m%d%H")
    input_date = model_run.strftime("%Y%m%d")
    input_year = model_run.strftime("%Y")
    forecast_hour = parser_args['forecast_hours']
    out_path = parser_args['base_path']

    model_run_str = model_run.strftime("%Y%m%d%H")
    model_run_str00 = model_run.strftime("%Y%m%d00")
    model_run_dir = model_run.strftime("%Y_%m_%d")
    ftime = model_run.strftime("%H")
    url = "https://www.usgodae.org/ftp/outgoing/fnmoc/models/navgem_0.5/"
    URL = f"{url}{input_year}/{model_run_str}"
    soup = create_soup(URL)
    grib_path = os.path.join(parser_args["base_path"], input_year, model_run_dir)
    os.makedirs(grib_path, exist_ok=True)
    files = url_search_nrl(soup, URL, model_run, forecast=forecast_hour, dest_path=grib_path)
                           
    LOG.info(files)
