"""Desktop side of the benchmark: runs desktop UVR's own separation (UVR.py) for each model on
the benchmark groups, timing each file and recording peak memory, into
<work>/desktop/<model>/<variant>/. Variants:
  cpu     every group on the CPU (UVR's default): quality, speed, memory
  gpu     group1 with "GPU Conversion" (Apple MPS): speed
  zeroed  VR only, while desktop still has the bug: every group on the CPU with
          cmb_spectrogram_to_wave's uninitialised array zeroed, as on the phone: quality
Settings are UVR's defaults except float WAV output and Demucs shifts = 0. Each model/variant
runs in its own process so its peak memory is its own; data.pkl is restored afterwards. The UVR
window is open while it runs. Resumable: finished model/variants are skipped.

    python desktop_bench.py <work dir> [model ...]
"""
import json
import os
import resource
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, HERE)
from models import MODELS

# Desktop VR left cmb_spectrogram_to_wave's array uninitialised until commit e972caa; with the fix
# in place a VR run already is the zeroed one
VR_BUG = 'np.ndarray(shape=' in open(os.path.join(REPO, 'lib_v5', 'spec_utils.py')).read()

def variants(key):
    return ['cpu', 'gpu'] + (['zeroed'] if MODELS[key][0] == 'VR_ARCH_PM' and VR_BUG else [])

def main():
    if sys.argv[1] == '--worker':
        return worker(*sys.argv[2:5])
    work = os.path.abspath(sys.argv[1])
    keys = sys.argv[2:] or list(MODELS)
    backup = os.path.join(work, 'data.pkl.backup')
    shutil.copy(os.path.join(REPO, 'data.pkl'), backup)
    try:
        for key in keys:
            for variant in variants(key):
                if os.path.exists(os.path.join(work, 'desktop', key, variant, 'result.json')):
                    continue
                subprocess.run([sys.executable, '-I', os.path.abspath(__file__), '--worker', work, key, variant])
    finally:
        shutil.copy(backup, os.path.join(REPO, 'data.pkl'))

def worker(work, key, variant):
    groups = json.load(open(os.path.join(work, 'groups.json')))
    files = [g for g in groups if variant != 'gpu' or g['id'] == 'group1']
    out_dir = os.path.join(work, 'desktop', key, variant)
    os.makedirs(out_dir, exist_ok=True)
    sys.path.insert(0, REPO)
    os.chdir(REPO)

    if variant == 'zeroed':
        import inspect, textwrap
        from lib_v5 import spec_utils
        src = textwrap.dedent(inspect.getsource(spec_utils.cmb_spectrogram_to_wave))
        assert src.count('np.ndarray(shape=') == 1
        exec(compile(src.replace('np.ndarray(shape=', 'np.zeros(shape='), spec_utils.__file__, 'exec'), spec_utils.__dict__)

    import UVR
    UVR.messagebox.showerror = lambda **kw: print('ERROR DIALOG:', kw.get('title'), '|', kw.get('message'), flush=True)
    UVR.messagebox.askyesno = lambda **kw: (print('QUESTION DIALOG:', kw.get('title'), '|', kw.get('message'), flush=True), False)[1]

    root = UVR.MainWindow()
    UVR.root = root
    root.update_checkbox_text()
    root.is_root_defined_var.set(True)
    root.is_check_splash = True
    root.update_idletasks()
    root.deiconify()

    method_name, wanted = MODELS[key][0], MODELS[key][1]
    method = getattr(UVR, method_name)
    runs = []

    def configure():
        root.load_saved_settings(UVR.DEFAULT_DATA, is_default_reset=True)
        root.wav_type_set_var.set('32-bit Float')
        root.shifts_var.set('0')
        root.is_gpu_conversion_var.set(variant == 'gpu')
        root.is_task_complete_var.set(False)
        root.export_path_var.set(out_dir)
        root.chosen_process_method_var.set(method)
        root.selection_action_process_method(method)
        var, option = {
            'VR_ARCH_PM': (root.vr_model_var, root.vr_model_Option),
            'MDX_ARCH_TYPE': (root.mdx_net_model_var, root.mdx_net_model_Option),
            'DEMUCS_ARCH_TYPE': (root.demucs_model_var, root.demucs_model_Option),
        }[method_name]
        matches = [v for v in option['values'] if v == wanted or v.split('|')[-1].strip() == wanted]
        if len(matches) != 1:
            print(f'{key}: "{wanted}" matches {matches}', flush=True)
            root.destroy()
            return
        var.set(matches[0])
        root.selection_action_models(matches[0])
        root.after(100, lambda: run_file(0))

    def run_file(i):
        if i == len(files):
            result = {'model': key, 'variant': variant, 'runs': runs,
                      'peak_rss_kb_end': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024}
            json.dump(result, open(os.path.join(out_dir, 'result.json'), 'w'), indent=2)
            print(f'{key}/{variant}: done {[round(r["wall_s"], 1) for r in runs]}', flush=True)
            root.destroy()
            return
        root.inputPaths = (os.path.join(work, 'groups', f"{files[i]['id']}.wav"),)
        root.update_inputPaths()
        t0 = time.time()
        root.process_initialize()
        root.after(200, lambda: poll(i, t0))

    def poll(i, t0):
        if root.thread_check(root.active_processing_thread):
            root.after(200, lambda: poll(i, t0))
            return
        wall = time.time() - t0
        # UVR's console shows queued messages every 100 ms, so let the last ones arrive before checking
        root.after(600, lambda: finish_file(i, wall))

    def finish_file(i, wall):
        ok = UVR.PROCESS_COMPLETE.strip() in root.command_Text.get('1.0', 'end')
        runs.append({'file': files[i]['id'], 'audio_s': files[i]['seconds'], 'wall_s': wall, 'ok': ok,
                     # macOS reports ru_maxrss in bytes; after the first file it is that run's peak
                     'peak_rss_kb_so_far': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024})
        if not ok:
            print(f'{key}/{variant} {files[i]["id"]} FAILED:', root.error_log_var.get()[:2000], flush=True)
        root.after(300, lambda: run_file(i + 1))

    root.after(2000, configure)
    root.mainloop()

if __name__ == '__main__':
    main()
