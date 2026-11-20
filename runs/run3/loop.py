import os
os.environ["K3_LOOP_CONFIG"] = "/root/harness/runs/run3/loop.json"
exec(compile(open("/root/harness/tools/loop_check.py").read(), "loop_check.py", "exec"))
