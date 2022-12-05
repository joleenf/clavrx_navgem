#!/bin/bash

#datecmd=`which date`
datecmd=date

export PS4='line:${LINENO} function:${FUNCNAME[0]:+${FUNCNAME[0]}() }cmd: ${BASH_COMMAND} \n result: '
set -x

# command line, can input number of days ago to retrieve old data as first argument
# goes_sync.sh 1

finish () {
    echo "exiting, removing lock" >> $LOG 2>&1
    $datecmd >> $LOG 2>&1
    rmdir $LOCK >> $LOG 2>&1
}

trap finish EXIT

# for some reason, this was a bad thing to remove, so keep set -ex
export data_date=$1
export run_hour=$2

declare -a data_vars=(pres_msl  pres  rltv_hum  air_temp  snw_dpth wnd_ucmp  wnd_vcmp  geop_ht  terr_ht  air_temp  vpr_pres  ice_cvrg)
declare -a data_vars=(pres)

test -n "$data_date" || export data_date=0
test -n "$run_hour" || export run_hour="00"

pd_regex="\b(${data_vars[0]}"
for product in "${data_vars[@]:1}"; do
    pd_regex=${pd_regex}\|$product
done
pd_regex="$pd_regex)\b"

echo $pd_regex

function do_sync {

    # start log entry
    $datecmd +"%D %H:%M:%S starting $0" >> $LOG 2>&1

    #make dest directory
    if mkdir -p $DEST >> $LOG 2>&1 ; then
      echo "mkdir $DEST succeeded, continuing" >> $LOG 2>&1
    fi

    #mirror data
    lftp -c "set ssl:verify-certificate no;
    open '$URL';
    lcd $DEST;
    cd $REMOTEPATH;
    mirror --only-newer --verbose=2 $lftp_mirror"
    cd $DEST
}

function retrieve_special {
     REMOTEPATH=ftp/outgoing/fnmoc/models/navgem_0.5/${YEAR}/${THISDATE}${pwat_hr}/
     file_pattern="$navgem_dir/US*GR1*${run_date}${pwat_hr}*${product_name}"
     lftp_mirror="--include-glob=US*.0018_0056_0[0-1]*\b(cape|prcp_h20) --exclude-glob=*.0018_0056_0[2-9]* --exclude-glob=*.0018_0056_1* --exclude-glob=\b(images|las|templates)\b"
     echo Retrieving special cases
     do_sync
}

function concat_gribs {
	# cat files into one grib
	navgem_dir=$1
	run_date=$2
	run_hr=$3
	product_name=$4 

	file_pattern="$navgem_dir/US*GR1*${run_date}${run_hr}*${product_name}"
	if compgen -G $file_pattern > /dev/null; then
		cat $file_pattern > $navgem_dir/navgem_${run_date}${run_hour}_${product_name}.grib
	else
		case "$product_name" in
			"prcp_h20"|"cape")
				case "$run_hour" in
					"00"|"06") pwat_hr="00";;
					"12"|"18") pwat_hr="12";;
				esac
				file_pattern="$navgem_dir/US*GR1*${run_date}${pwat_hr}*${product_name}"
				if !(compgen -G $file_pattern) > /dev/null; then
					retrieve_special $file_pattern
				else
					cat $file_pattern > $navgem_dir/navgem_${run_date}${pwat_hr}_${product_name}.grib
				
				fi

				if !(compgen -G $file_pattern) > /dev/null; then
				    echo "ERROR: No $product_name found for ${run_date}${pwat_hr} from US*${run_date}00*${product_name}"
				fi
				;;
			*)  echo "ERROR: No $product_name found for ${run_date}${run_hr} from US*${run_date}00*${product_name}"
		              exit 1;;
	                esac
	fi
}


DIR=$HOME
YEAR=$(${datecmd} +%Y -d "${data_date}")
THISDATE=$(${datecmd} +"%Y%m%d" -d "${data_date}")
TODAY=$(${datecmd} +"%Y_%m_%d" -d "${data_date}")

LOCK=$DIR/.${THISDATE}.lock
LOG=$DIR/logs/nrl_${THISDATE}_sync.log
DEST=/data/Personal/joleenf/navgem/$YEAR/${TODAY}/nrl_orig

mkdir -p $DEST
#make a lock directory
if mkdir $LOCK >> $LOG 2>&1 ; then
  echo "mkdir $LOCK succeeded, continuing" >> $LOG 2>&1
else
  echo "Lock (mkdir $LOCK) failed, NRL_NAVGEM_${run_hr}_sync assumed running, exiting" >> $LOG 2>&1
  exit 0
fi

URL=https://www.usgodae.org/
REMOTEPATH=ftp/outgoing/fnmoc/models/navgem_0.5/${YEAR}/${THISDATE}${run_hour}/
# include glob to capture forecasts up to 18Z but exclude all forecasts 21Z-144Z.
lftp_mirror="--include-glob=US*.0018_0056_0[0-1]* --exclude-glob=*.0018_0056_0[2-9]* --exclude-glob=*.0018_0056_1*"
do_sync

for product in "${data_vars[@]}";do
    concat_gribs $DEST $THISDATE $run_hour $product
done

exit
