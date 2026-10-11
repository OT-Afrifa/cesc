# Caveats and settled decisions

These were settled by the authors during the study. The package
encodes them; this page says why.

theta_e, not theta_v and not dry theta. Potential instability is a
saturation criterion: the latent-heat contribution carried by theta_e
is what makes a moist layer unstable when lifted to saturation
(Schultz and Schumacher 1999; Bolton 1980 for the computation). The
preflight rejects profiles whose magnitude looks like dry theta.

Reflectivity level. Default 1 km AGL for continuity with radar
climatologies; the column maximum gives essentially the same objects
for shallow cold-season cells and is a fine substitute. The lowest
model level is not offered: over complex terrain it is divorced from
tropospheric dynamics and shrinks objects to the precipitation
footprint, which would also require retuning box_sizes.

Rain-free pixels inside objects are by design. The object outlines the
cell aloft; over dry valleys part of the falling ice sublimates below
cloud base. These pixels add area and no precipitation, so they move
area fractions and not precipitation fractions.

bg_radius is display and watershed seeding only. It never enters the
detection thresholds. Conflating the two is a known misreading.

Instantaneous rates, not accumulations. An hourly accumulation trails
moving cells by several cell diameters. In the reference case the
event precipitation fraction rose from 10.6% to 12.3% and the peak
hour from 15.8% to 28.0% when the displaced accumulation was replaced
with the instantaneous rate. The preflight warns on accumulation-like
inputs. Reconstructing an instantaneous rate from hydrometeor fields
is scheme-specific and lives outside the core package; for the
Thompson scheme the reconstruction used in the study recovered 90% of
the model's monthly accumulation, with the deficit systematic in the
offline flux computation and cancelling in precipitation fractions.

No tracking. Hourly fields cannot follow 20-40 minute cells advecting
at 15-30 m/s. Objects are identified per timestep.
cesc.contrib.linker links objects between consecutive hours by
expected advection displacement; it is experimental and qualitative.

Objects do not coincide with surface-rate maxima. Instantaneous
surface rates inside cold-season CESC overlap the surrounding
stratiform; the cells separate in reflectivity structure and vertical
motion, not surface rate.

## What the theta_e preflight does and does not catch

The preflight validates theta_e only in the layer the algorithm reads,
below max_pi_height (5000 m by default). This matters because above the
boundary layer theta_e reduces to dry theta and rises steeply into the
stratosphere: in a full 80-level WRF column, values above 12 km
routinely exceed 500 K, and roughly 5% of an interior-Pacific-Northwest
winter field sits above 425 K purely from those upper levels. Those
values are physical and never enter the scan, so the check ignores
them and reports the statistics of the scanned layer.

Within that layer the check is a magnitude bracket: it fails a field
whose 0-5 km median falls outside about 250-380 K, which catches a
Celsius-scale field or one shifted far from the physical band. It does
not attempt to separate theta_e from dry theta by profile shape. In a
well-mixed cold-season boundary layer both rise smoothly by 10-15 K
over the lowest 5 km and differ mainly by a roughly 10 K offset that is
within the range of surface-temperature variation, so a shape test is
unreliable there and was removed rather than left to fail on good data.
The consequence is a known blind spot: a dry-theta field whose 0-5 km
values happen to overlap physical theta_e will pass. Supplying theta_e
computed with cesc.thermo.theta_e_bolton, or from a source known to be
equivalent potential temperature, is the guard the preflight cannot
fully provide.

## Running a month: batch, not a login node

Run `run_wrf_month` as a batch job on a compute node. Derecho login nodes
reap long-running CPU-heavy processes, and a 672-timestep month is
exactly that: the first February 2017 attempt was launched with `nohup`
on a login node and was killed after about 100 timesteps, leaving a
valid but silently short output file. `examples/run_month.pbs` submits it
correctly.

The runner writes a part file every `checkpoint_every` timesteps into
`<out_nc_path>.parts/` and clears its in-memory accumulator, so memory
stays flat rather than growing with the month (the earlier design held
every timestep in RAM, about 19 MB per step on a 540 x 810 grid, which
reaches roughly 13 GB accumulated and 25 GB at the final write for a full
month). Part files are renamed into place atomically, so an interrupted
run never leaves a half-written part.

Interrupted runs resume. Resubmit the same job with `resume=True` (the
default): timesteps already present in the part files are skipped and the
run continues from the first missing one, producing a result identical to
an uninterrupted run. Only the final stitch marks the output
`cesc_complete = "yes"`; open results with `cesc.open_output`, which
refuses a partial file rather than handing back a short record.

A note on the batch script and the module system: do not call
`module --force purge` before `module load conda`. Purging removes the
modules that make `conda` findable, so the load then fails with "exist
but cannot be loaded". `examples/run_month.pbs` sidesteps this by not
using the module system at all: it invokes the environment's python by
absolute path. A Python interpreter called by absolute path resolves its
own prefix and site-packages regardless of `PATH` or activation state, so
no `conda activate` is needed. Run `bash examples/check_batch_env.sh` on
a login node first; it exercises the same environment setup and imports
in a few seconds, so a missing dependency surfaces before the job queues.

The parts directory is working state, not output. Once a run finishes,
the stitched NetCDF is re-opened and checked (timestep count against the
full request, the completeness flag, and the object CSV row count) and
only then is `<out_nc_path>.parts` removed. It roughly doubles the disk
a month occupies, so keeping it after success buys nothing. Parts of an
incomplete run are always kept, because resubmitting resumes from them.
Pass `keep_parts=True` to retain them regardless.