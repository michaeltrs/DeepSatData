import datetime
import os
import re

def get_doy(date):
    Y = date[:4]
    m = date[4:6]
    d = date[6:]
    date = "%s.%s.%s" % (Y, m, d)
    dt = datetime.datetime.strptime(date, '%Y.%m.%d')
    return dt.timetuple().tm_yday


def get_date(day):
    """
    :param day: day of the year [0, 365]
    :return: sting day_of_month-month, ie. "3-Jul"
    """
    if day < 31:
        m = "Jan"
        d = day
    elif day < 59:
        m = "Feb"
        d = day - 31
    elif day < 90:
        m = "Mar"
        d = day - 59
    elif day < 120:
        m = "Apr"
        d = day - 90
    elif day < 151:
        m = "May"
        d = day - 120
    elif day < 181:
        m = "Jun"
        d = day - 151
    elif day < 212:
        m = "Jul"
        d = day - 181
    elif day < 243:
        m = "Aug"
        d = day - 212
    elif day < 273:
        m = "Sep"
        d = day - 243
    elif day < 304:
        m = "Oct"
        d = day - 273
    elif day < 334:
        m = "Nov"
        d = day - 304
    else:
        m = "Dec"
        d = day - 334
    return "%d-%s" % (d, m)


def get_paths(root_dir, pattern, save_name=None, relative=True):
    files = glob(os.path.join(root_dir, pattern))
    N = len(root_dir.split("/"))
    if relative:
        # base = "/".join(pattern.split("/")[:-1])
        files = ["/".join(x.split("/")[N:]) for x in files]
    print("%d files found matching %s" % (len(files), pattern))
    if save_name:
        # check if abs path
        if not os.path.exists(save_name):
            save_name = os.path.join(root_dir, save_name)
        pd.DataFrame(files).to_csv(save_name, header=None, index=False)
    else:
        return files


def get_unique_vals(path, col, header=None, name_fn=None):
    data = pd.read_csv(path, header=header)
    data = data[col]
    if name_fn:
        data = data.apply(name_fn)
    return data.value_counts()


def get_lat_lon(loc, loc_type="meters"):
    if loc_type == "meters":
        lat = 111319.488
        lon = 111120.0
        return loc[0] / lat, loc[1] / lon
