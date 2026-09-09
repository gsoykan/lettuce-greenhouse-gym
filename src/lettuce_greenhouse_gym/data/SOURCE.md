# Packaged weather data

## `outdoorWeatherWurGlas2014.csv`

Outdoor weather measured at the **WUR Glas** reference greenhouse facility, **Bleiswijk, the
Netherlands**, during 2014. Shipped **unmodified**, exactly as obtained.

This is the trace the van Henten lettuce RL/MPC benchmark is defined on: 40-day seasons beginning
9 February 2014 at a 30-minute control step.

### Attribution

The facility and the experiments during which these measurements were recorded are described in:

> Kempkes, F. L. K., Janse, J., & Hemming, S. (2014). *Greenhouse concept with high insulating
> double glass with coatings and new climate control strategies; from design to results from tomato
> experiments.* Acta Horticulturae, (1037), 83–92.
> https://doi.org/10.17660/ActaHortic.2014.1037.6

Please cite the reference above in any work that uses this trace.

### Format

Headerless CSV, 93,311 rows, one row every **300 s** (5 minutes), spanning **10 January to
29 November 2014** (324 days).

| col | name     | meaning                     | unit |
|-----|----------|-----------------------------|------|
| 0   | `time`   | seconds from the file start | s    |
| 1   | `Io`     | outdoor global radiation    | W/m² |
| 2   | `To`     | outdoor air temperature     | °C   |
| 3   | `RH`     | outdoor relative humidity    | %    |
| 4   | `Vo`     | outdoor wind speed          | m/s  |
| 5   | `CO2ppm` | outdoor CO₂ concentration   | ppm  |

`WeatherSeries.from_csv` converts `RH → vapour density` and `CO2ppm → CO₂ density` with the
package's own conversions, drops wind (the four-state model has no wind term), and yields the
disturbance vector `v = [rad, out_co2, out_temp, out_vapor]`.

### A note on the epoch

The `time` column counts **seconds from the file's own start**, not a calendar date — the export
that produced this CSV replaced the original MATLAB datenums with elapsed seconds and so discarded
the calendar. The first sample is **10 January 2014** (day of year 10), recovered from the datenums
in the original `.mat` artifact; the loader supplies it as `epoch_day` so that `start_day` means a
day of the year rather than an offset into a particular file. Sources beginning on different dates
(KNMI years start on 1 January) therefore line up without per-file corrections.
