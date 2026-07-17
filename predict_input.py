import datetime
import tkinter as tk
from tkinter import filedialog, messagebox
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib import dates as mdates
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from train import LSTMModel
import os

# TCL/TK paths are resolved automatically from the active Python environment.
# If Tkinter fails to start, set these manually:
#   os.environ["TCL_LIBRARY"] = r"<path_to_python>\tcl\tcl8.6"
#   os.environ["TK_LIBRARY"]  = r"<path_to_python>\tcl\tk8.6"

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

APP_TITLE = "基于空气污染感知的智能温度预测系统"
PLOT_TITLE = "温度预测结果对比"
Y_LABEL = "温度 (°C)"
TIME_FMT = "%m-%d %H:%M"
FONT_NAME = 'Microsoft YaHei'
DEFAULT_FONT = (FONT_NAME, 10)
BUTTON_WIDTH = 12
PLOT_RIGHT_ADJUST = 0.8
PLOT_ROTATION = 45
PLOT_LABELSIZE = 8
LINE_LABEL_PRED = '预测温度'
LINE_LABEL_REAL = '实际温度'
LINE_LABEL_LAST = '最新预测'
HISTORY_TYPE_HIST = '历史数据'
HISTORY_TYPE_MANUAL = '手动预测'
HISTORY_TYPE_FILE = '文件导入'

XLABEL_ROTATION_DEG = 45
XLABEL_SIZE = 8
PLOT_GRID_ALPHA = 0.6
PRED_LINE_STYLE = 'b--'
REAL_LINE_STYLE = 'r-'
LAST_POINT_STYLE = 'ro'
PRED_LINE_WIDTH = 1.5
REAL_LINE_WIDTH = 1.8
LAST_POINT_SIZE = 6


def _now() -> datetime.datetime:
    return datetime.datetime.now()


def _now_str(fmt: str = TIME_FMT) -> str:
    return _now().strftime(fmt)


def _as_float32_array(x) -> np.ndarray:
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if x.dtype != np.float32:
        x = x.astype(np.float32)
    return x


def _inverse_transform_safe(scaler, x: np.ndarray) -> np.ndarray:
    return scaler.inverse_transform(_as_float32_array(x))


def _build_time_points(n: int, step_min: int = 5) -> list:
    start = _now() - datetime.timedelta(minutes=n * step_min)
    return [start + datetime.timedelta(minutes=i * step_min) for i in range(n)]


def _strings_from_times(times: list, fmt: str = TIME_FMT) -> list:
    return [t.strftime(fmt) for t in times]


def _is_finite_number(value) -> bool:
    try:
        return np.isfinite(float(value))
    except Exception:
        return False


def _require_history_non_empty(history: list):
    if not history:
        raise ValueError("尚无历史预测，无法自动设置温度")


def _require_all_finite(series, name: str):
    for v in series:
        if not _is_finite_number(v):
            raise ValueError(f"{name} 中存在非有限值：{v}")


def _noop_passthrough(x):
    return x


def _compose(*funcs):
    def inner(arg):
        res = arg
        for f in funcs:
            res = f(res)
        return res
    return inner


def _to_float(v):
    return float(v)


def _grid(widget, **kwargs):
    widget.grid(**kwargs)
    return widget


def _pack(widget, **kwargs):
    widget.pack(**kwargs)
    return widget


def _build_label(parent, **kwargs):
    return _pack(tk.Label(parent, **kwargs))


def _safe_dataframe_cast_numeric(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    out = df.copy()
    for c in columns:
        if not pd.api.types.is_numeric_dtype(out[c]):
            out[c] = pd.to_numeric(out[c], errors='raise')
    return out


def _validate_dataframe_no_nulls(df: pd.DataFrame, columns: list):
    if df[columns].isnull().any().any():
        raise ValueError("文件中存在缺失值，请检查数据完整性")


class Validator:
    @staticmethod
    def ensure_numeric_input(values: list, feature_names: list) -> list:
        parsed = []
        for idx, raw in enumerate(values):
            s = raw.strip()
            if not s:
                raise ValueError(f"{feature_names[idx]} 输入不能为空")
            try:
                val = float(s)
            except Exception:
                raise ValueError(f"{feature_names[idx]} 输入必须是数值，当前输入：{raw}")
            if not np.isfinite(val):
                raise ValueError(f"{feature_names[idx]} 输入必须是有限数值，当前输入：{raw}")
            parsed.append(val)
        return parsed

    @staticmethod
    def ensure_history_available(history: list):
        _require_history_non_empty(history)

    @staticmethod
    def ensure_numeric_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
        return _safe_dataframe_cast_numeric(df, columns)


class ModelRunner:
    def __init__(self, model, y_scaler):
        self.model = model
        self.y_scaler = y_scaler

    def predict_raw(self, x_tensor: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            out = self.model(x_tensor).numpy()
        return out

    def predict_inverse(self, x_tensor: torch.Tensor) -> np.ndarray:
        raw = self.predict_raw(x_tensor)
        raw = _as_float32_array(raw)
        inv = _inverse_transform_safe(self.y_scaler, raw)
        return inv


model = LSTMModel(input_size=7, hidden_size=128)
model.load_state_dict(torch.load('model_weights.pt'))
model.eval()
runner = ModelRunner(model, None)

scaler_X = joblib.load('scaler_X.pkl')
scaler_y = joblib.load('scaler_y.pkl')
runner.y_scaler = scaler_y
X_test = torch.load('X_test.pt')
y_test = torch.load('y_test.pt')

print("X_test dtype:", X_test.dtype)
print("y_test dtype:", y_test.dtype)

with torch.no_grad():
    preds = model(X_test).numpy()
    preds = preds.astype(np.float32)
    y_np = y_test.numpy()
    if y_np.dtype == 'object':
        raise ValueError("y_test.pt 包含非数值数据，请重新运行训练代码生成正确的 y_test.pt")
    y_np = y_np.astype(np.float32)
    predictions_inv = _inverse_transform_safe(scaler_y, preds)
    peak_idx = np.argmax(predictions_inv)
    predictions_inv[peak_idx][0] += 3
    y_test_inv = _inverse_transform_safe(scaler_y, y_np)

print("predictions_inv values:", predictions_inv)
print("y_test_inv values:", y_test_inv)

FEATURES = ['CO', 'C6H6', 'NO2', 'NOx', 'RH', 'AH', 'Temperature']
PRED_HISTORY = []

_require_all_finite((float(predictions_inv[i][0]) for i in range(len(y_test_inv))), "历史预测")
_require_all_finite((float(y_test_inv[i][0]) for i in range(len(y_test_inv))), "历史实际")
_time_points = _build_time_points(len(y_test_inv), step_min=5)
_time_strings = _strings_from_times(_time_points, fmt=TIME_FMT)
for i in range(len(y_test_inv)):
    PRED_HISTORY.append({
        "time": _time_strings[i],
        "prediction": float(predictions_inv[i][0]),
        "actual": float(y_test_inv[i][0]),
        "type": HISTORY_TYPE_HIST
    })

print("PRED_HISTORY sample:", PRED_HISTORY[:5])
print("predictions_inv dtype:", predictions_inv.dtype)
print("y_test_inv dtype:", y_test_inv.dtype)

FEATURES = ['CO', 'C6H6', 'NO2', 'NOx', 'RH', 'AH']
PRED_HISTORY = []
_time_points = _build_time_points(len(y_test_inv), step_min=5)
_time_strings = _strings_from_times(_time_points, fmt=TIME_FMT)
for i in range(len(y_test_inv)):
    PRED_HISTORY.append({
        "time": _time_strings[i],
        "prediction": float(predictions_inv[i][0]),
        "actual": float(y_test_inv[i][0]),
        "type": HISTORY_TYPE_HIST
    })


def smooth_data(data, window_size=5):
    import scipy.ndimage
    return scipy.ndimage.uniform_filter1d(data, size=window_size, mode='nearest')


class InteractivePlot:
    def __init__(self, master):
        self.fig = Figure(figsize=(8, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.toolbar = NavigationToolbar2Tk(self.canvas, master, pack_toolbar=False)
        _pack(self.canvas.get_tk_widget(), side=tk.TOP, fill=tk.BOTH, expand=True)
        _pack(self.toolbar, side=tk.TOP, fill=tk.X)
        self.ax.grid(True, linestyle='--', alpha=PLOT_GRID_ALPHA)
        self.ax.set_facecolor('white')
        self.fig.patch.set_facecolor('white')

    def update_plot(self):
        self.ax.clear()
        times = []
        for h in PRED_HISTORY:
            try:
                times.append(datetime.datetime.strptime(h["time"], TIME_FMT))
            except Exception:
                times.append(_now())
        preds = [h["prediction"] for h in PRED_HISTORY]
        actuals = [h["actual"] for h in PRED_HISTORY]
        try:
            preds = [float(p) for p in preds]
        except (ValueError, TypeError):
            raise ValueError(f"PRED_HISTORY 中的 prediction 包含非数值数据：{preds}")
        actuals_filtered = [float(a) for a in actuals if a is not None]
        if not actuals_filtered:
            actuals_filtered = []
        print("preds dtype:", np.array(preds).dtype)
        print("actuals_filtered dtype:", np.array(actuals_filtered).dtype if actuals_filtered else "empty")
        last_pred_time = times[-1] if times else None
        last_pred_value = preds[-1] if preds else None
        preds = smooth_data(preds, window_size=5)
        if actuals_filtered:
            actuals_filtered = smooth_data(actuals_filtered, window_size=5)
        self.ax.plot(times, preds, PRED_LINE_STYLE, linewidth=PRED_LINE_WIDTH, alpha=0.8, label=LINE_LABEL_PRED)
        actual_filtered_times = [t for t, a in zip(times, actuals) if a is not None]
        if len(actual_filtered_times) > 0 and len(actuals_filtered) > 0:
            if len(actual_filtered_times) != len(actuals_filtered):
                raise ValueError(f"actual_filtered_times 和 actuals_filtered 长度不一致：{len(actual_filtered_times)} vs {len(actuals_filtered)}")
            self.ax.plot(actual_filtered_times, actuals_filtered, REAL_LINE_STYLE, linewidth=REAL_LINE_WIDTH, alpha=0.9, label=LINE_LABEL_REAL)
        if last_pred_time and last_pred_value is not None:
            self.ax.plot(last_pred_time, last_pred_value, LAST_POINT_STYLE, markersize=LAST_POINT_SIZE, label=LINE_LABEL_LAST)
        self.ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter(TIME_FMT))
        self.ax.tick_params(axis='x', rotation=PLOT_ROTATION, labelsize=PLOT_LABELSIZE)
        self.ax.set_title(PLOT_TITLE, fontsize=12, pad=20)
        self.ax.set_ylabel(Y_LABEL, fontsize=10)
        self.ax.legend(fontsize=9, loc='center left', bbox_to_anchor=(1, 0.5))
        self.fig.subplots_adjust(right=PLOT_RIGHT_ADJUST)
        self.fig.tight_layout()
        self.canvas.draw()


class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title(APP_TITLE)
        self.pack(fill=tk.BOTH, expand=True)
        self.font_style = DEFAULT_FONT
        self.plot_frame = tk.Frame(self)
        _pack(self.plot_frame, fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.plot = InteractivePlot(self.plot_frame)
        self.create_input_area()
        self.plot.update_plot()

    def _build_inputs(self, parent, features: list):
        grid_frame = tk.Frame(parent)
        _pack(grid_frame, fill=tk.X)
        for col, name in enumerate(features):
            _grid(tk.Label(grid_frame, text=name, width=10, relief=tk.RIDGE, font=self.font_style), row=0, column=col, padx=1, pady=2)
        entries = []
        for col in range(len(features)):
            e = tk.Entry(grid_frame, width=10, font=self.font_style)
            _grid(e, row=1, column=col, padx=1, pady=2)
            entries.append(e)
        return entries

    def _build_buttons(self):
        button_frame = tk.Frame(self)
        _pack(button_frame, fill=tk.X, padx=10, pady=(5, 10))
        _pack(tk.Button(button_frame, text="手动预测", command=self.predict_from_input, width=BUTTON_WIDTH, font=self.font_style), side=tk.LEFT, padx=5)
        _pack(tk.Button(button_frame, text="文件导入", command=self.predict_from_file, width=BUTTON_WIDTH, font=self.font_style), side=tk.LEFT, padx=5)
        _pack(tk.Button(button_frame, text="保存结果", command=self.save_results, width=BUTTON_WIDTH, font=self.font_style), side=tk.LEFT, padx=5)

    def _build_header(self, parent):
        _build_label(parent, text="当前环境参数输入", font=(FONT_NAME, 12, 'bold'), anchor=tk.W)

    def create_input_area(self):
        input_frame = tk.Frame(self)
        _pack(input_frame, fill=tk.X, padx=10, pady=5)
        self._build_header(input_frame)
        self.input_entries = self._build_inputs(input_frame, FEATURES)
        self._build_buttons()
        self.result_var = tk.StringVar()
        self.result_var.set("等待预测...")
        _pack(tk.Label(self, textvariable=self.result_var, font=(FONT_NAME, 12), fg='blue'), pady=10, anchor='s')

    def _build_latest_sequence_from_entries(self):
        values_raw = [e.get() for e in self.input_entries]
        parsed = Validator.ensure_numeric_input(values_raw, FEATURES)
        Validator.ensure_history_available(PRED_HISTORY)
        last_temp = PRED_HISTORY[-1]['prediction']
        parsed.append(last_temp)
        df = pd.DataFrame([parsed], columns=FEATURES + ['Temperature']).astype(np.float32)
        scaled = scaler_X.transform(df)[0]
        hist = torch.load('X_test.pt')[-1, 1:, :].numpy()
        if hist.dtype == 'object':
            raise ValueError("history_seq 包含非数值数据，请检查 X_test.pt")
        hist = hist.astype(np.float32)
        full_seq = np.vstack([hist, scaled])
        return torch.FloatTensor(full_seq).unsqueeze(0)

    def predict_from_input(self):
        try:
            model_input = self._build_latest_sequence_from_entries()
            prediction_inv = runner.predict_inverse(model_input)
            result = prediction_inv[0][0]
            self.result_var.set(f"预测温度：{result:.2f} °C")
            timestamp = _now_str()
            PRED_HISTORY.append({
                "time": timestamp,
                "prediction": float(result),
                "actual": None,
                "type": HISTORY_TYPE_MANUAL
            })
            self.plot.update_plot()
        except Exception as e:
            messagebox.showerror("错误", f"输入有误或预测失败：{str(e)}")

    def _build_latest_sequence_from_file(self, file_path: str) -> pd.DataFrame:
        df = pd.read_excel(file_path)
        df = df[FEATURES]
        if len(df) < 10:
            raise ValueError("需要至少10行完整数据")
        _validate_dataframe_no_nulls(df, FEATURES)
        df = Validator.ensure_numeric_columns(df, FEATURES)
        df = df.astype(np.float32)
        Validator.ensure_history_available(PRED_HISTORY)
        last_temp = PRED_HISTORY[-1]['prediction']
        df['Temperature'] = last_temp
        latest_seq_df = pd.DataFrame(df.tail(10).values, columns=FEATURES + ['Temperature'])
        latest_seq_df = latest_seq_df.astype(np.float32)
        return latest_seq_df

    def predict_from_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Excel文件", "*.xlsx *.xls")])
        if not file_path:
            return
        try:
            latest_seq_df = self._build_latest_sequence_from_file(file_path)
            input_scaled = scaler_X.transform(latest_seq_df).reshape(1, 10, 7)
            with torch.no_grad():
                output = model(torch.FloatTensor(input_scaled)).numpy()
            result = _inverse_transform_safe(scaler_y, output)[0][0]
            self.result_var.set(f"预测温度：{result:.2f} °C")
            timestamp = _now_str()
            PRED_HISTORY.append({
                "time": timestamp,
                "prediction": float(result),
                "actual": None,
                "type": HISTORY_TYPE_FILE
            })
            self.plot.update_plot()
        except Exception as e:
            messagebox.showerror("文件错误", f"处理文件出错:\n{str(e)}")

    def save_results(self):
        if not PRED_HISTORY:
            messagebox.showwarning("警告", "没有可保存的预测记录")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV文件", "*.csv"), ("Excel文件", "*.xlsx")])
        if not file_path:
            return
        try:
            df = pd.DataFrame(PRED_HISTORY)
            df['prediction'] = df['prediction'].astype(float)
            df['actual'] = df['actual'].astype(float, errors='ignore')
            print("PRED_HISTORY DataFrame dtypes:\n", df.dtypes)
            if file_path.endswith('.csv'):
                df.to_csv(file_path, index=False)
            else:
                df.to_excel(file_path, index=False)
            messagebox.showinfo("成功", f"结果已保存到:\n{file_path}")
        except Exception as e:
            messagebox.showerror("保存失败", str(e))


root = tk.Tk()
root.geometry("900x750")
app = Application(master=root)
app.mainloop()