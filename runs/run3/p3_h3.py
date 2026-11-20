import os
os.environ["K3_PROBE_CONFIG"] = "/root/harness/runs/run3/p3_h3.json"
exec(compile(open("/root/harness/tools/quick_probe.py").read(), "quick_probe.py", "exec"))
