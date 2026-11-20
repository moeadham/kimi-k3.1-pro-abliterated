import os
os.environ["K3_PE_CONFIG"] = "/root/harness/runs/run3/pe_build.json"
exec(compile(open("/root/harness/tools/pe_capture.py").read(), "pe_capture.py", "exec"))
