"""Desktop side: runs desktop UVR's own separation (UVR.py's process_start: CPU, original
models, 32-bit float WAV out) on every clip into <work>/desktop/<model>/. Settings are UVR's
defaults except float WAV output and Demucs shifts = 0 (see phone_run.py). The UVR window is
open while it runs; data.pkl (UVR's saved settings) is restored afterwards.

    python desktop_refs.py <work dir> [model ...]
"""
import json
import os
import shutil
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))

# key -> (UVR process method attribute, model name as UVR lists it)
MODELS = {
    'vr13': ('VR_ARCH_PM', '13_SP-UVR-4B-44100-1'),
    'mdx_kim2': ('MDX_ARCH_TYPE', 'Kim Vocal 2'),
    'htdemucs': ('DEMUCS_ARCH_TYPE', 'htdemucs'),
}

def main():
    work = os.path.abspath(sys.argv[1])
    keys = sys.argv[2:] or list(MODELS)
    clips = json.load(open(os.path.join(work, 'clips.json')))
    inputs = tuple(os.path.join(work, 'clips', f"{c['id']}.wav") for c in clips)

    sys.path.insert(0, REPO)
    os.chdir(REPO)
    backup = os.path.join(work, 'data.pkl.backup')
    shutil.copy('data.pkl', backup)
    try:
        run(work, keys, inputs)
    finally:
        shutil.copy(backup, 'data.pkl')

def run(work, keys, inputs):
    import UVR

    # Unattended: report dialogs instead of waiting on them
    UVR.messagebox.showerror = lambda **kw: print('ERROR DIALOG:', kw.get('title'), '|', kw.get('message'), flush=True)
    UVR.messagebox.askyesno = lambda **kw: (print('QUESTION DIALOG:', kw.get('title'), '|', kw.get('message'), flush=True), False)[1]

    root = UVR.MainWindow()
    UVR.root = root
    root.update_checkbox_text()
    root.is_root_defined_var.set(True)
    root.is_check_splash = True
    root.update_idletasks()
    root.deiconify()

    pending = list(keys)
    state = {}

    def start_next():
        if not pending:
            root.destroy()
            return
        key = pending.pop(0)
        method_name, wanted = MODELS[key]
        method = getattr(UVR, method_name)
        out = os.path.join(work, 'desktop', key)
        os.makedirs(out, exist_ok=True)

        root.load_saved_settings(UVR.DEFAULT_DATA, is_default_reset=True)
        root.wav_type_set_var.set('32-bit Float')
        root.shifts_var.set('0')
        root.is_gpu_conversion_var.set(False)
        root.is_task_complete_var.set(False)
        root.export_path_var.set(out)
        root.chosen_process_method_var.set(method)
        root.selection_action_process_method(method)

        var, option = {
            'VR_ARCH_PM': (root.vr_model_var, root.vr_model_Option),
            'MDX_ARCH_TYPE': (root.mdx_net_model_var, root.mdx_net_model_Option),
            'DEMUCS_ARCH_TYPE': (root.demucs_model_var, root.demucs_model_Option),
        }[method_name]
        matches = [v for v in option['values'] if v == wanted or v.split('|')[-1].strip() == wanted]
        if len(matches) != 1:
            print(f'{key}: "{wanted}" matches {matches} in {list(option["values"])}', flush=True)
            root.after(10, start_next)
            return
        var.set(matches[0])
        root.selection_action_models(matches[0])

        root.inputPaths = inputs
        root.update_inputPaths()
        print(f'{key}: {matches[0]} on {len(inputs)} clips → {out}', flush=True)
        state.update(key=key, t0=time.time())
        root.process_initialize()
        root.after(1000, poll)

    def poll():
        if root.thread_check(root.active_processing_thread):
            root.after(1000, poll)
            return
        console = root.command_Text.get('1.0', 'end')
        ok = UVR.PROCESS_COMPLETE.strip() in console
        print(f"{state['key']}: {'done' if ok else 'FAILED'} in {time.time() - state['t0']:.0f} s", flush=True)
        if not ok:
            print(console[-3000:], root.error_log_var.get()[:3000], sep='\n', flush=True)
        root.after(500, start_next)

    root.after(2000, start_next)
    root.mainloop()

if __name__ == '__main__':
    main()
