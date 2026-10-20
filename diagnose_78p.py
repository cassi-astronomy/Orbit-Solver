import json
import sys

import requests

from orbit_app.mpc import MpcClient
from orbit_app.mpc_orbit import parse_mpc_orbit
from orbit_app.orbit_fit import _observing_apparition_groups, fit_orbit


sys.stdout.reconfigure(encoding="utf-8", errors="replace")

session = requests.Session()
session.headers.update({"User-Agent": "OrbitSolver/1.0"})
response = session.get(
    "https://data.minorplanetcenter.net/api/get-orb",
    json={"desig": "78P"}, timeout=120,
)
response.raise_for_status()
initial = parse_mpc_orbit(response.json(), "78P")
observations = MpcClient(timeout=180).observations("78P")
groups = [
    group for group in _observing_apparition_groups(observations, initial)
    if len(group) >= 3
]
print("INPUT", len(observations), flush=True)
print("SEED", initial.epoch_jd, initial.nongravitational_parameters, flush=True)
for group in groups:
    print("GROUP", group[0].time, group[-1].time, len(group), flush=True)

parameters = {
    "A1": float((initial.nongravitational_parameters or {}).get("A1", 0.0)),
    "A2": float((initial.nongravitational_parameters or {}).get("A2", 0.0)),
    "A3": float((initial.nongravitational_parameters or {}).get("A3", 0.0)),
    "DT": float((initial.nongravitational_parameters or {}).get("DT", 0.0)),
}


def progress(label, step, total):
    print(f"[{step}/{total}] {label}", flush=True)


solution, rms, used = fit_orbit(
    observations, initial, nongravitational=parameters,
    progress_callback=progress,
)
print("RESULT", rms, solution.fit_normalized_rms, used, flush=True)
print("NG", json.dumps(solution.nongravitational_parameters), flush=True)
print("NOTE", solution.fit_model_note, flush=True)
for record in solution.apparition_linkage or ():
    print("LINK", record, flush=True)
