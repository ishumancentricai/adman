#!/usr/bin/env python3
"""
Small Tkinter app:
- Global HEADERS constant defines column headers
- For each header you can enter a value and choose a datatype
- Buttons: Reset, Save to CSV
- Scrollable form; mouse wheel scrolls the form
"""

import csv
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime

HEADERS = [
    "id",
    "startdate",
    "submitdate",
    "lastpage",
    "PID",
    "STUDYID",
    "SESSIONID",
    "gender",
    "age",
    "birthCountry",
    "ethnicity",
    "familyOrigin",
    "originCountry",
    "originCountry[other]",
    "residenceCountry",
    "residenceCity",
    "occupationType",
    "occupationTitle",
    "incomeClass",
    "relationshipStatus",
    "relationshipStatus[other]",
    "childrenNumber",
    "childrenAge[INFA]",
    "childrenAge[YCHIL]",
    "childrenAge[CHIL]",
    "childrenAge[TEEN]",
    "childrenAge[YADUL]",
    "childrenAge[ADUL]",
    "regularActivities",
    "housingType",
    "housingType[other]",
    "residenceArea",
    "climate",
    "petOwnership",
    "petType",
    "hairStyle",
    "hairStyle[other]",
    "hairColor",
    "hairColor[other]",
    "beardStyle",
    "bodyType",
    "glassesWearer",
    "colorPreference",
    "stylePreference[MIN]",
    "stylePreference[STREET]",
    "stylePreference[CLASS]",
    "stylePreference[TREND]",
    "stylePreference[OUTD]",
]

DATATYPES = ["string", "int", "float", "date", "bool"]

def try_parse_date(text: str):
    if not text:
        return ""
    text = text.strip()
    try:
        d = datetime.fromisoformat(text)
        return d.date().isoformat()
    except Exception:
        pass
    fmts = [
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%Y/%m/%d",
        "%m/%d/%Y",
        "%d.%m.%Y",
        "%Y-%m-%d %H:%M:%S",
    ]
    for f in fmts:
        try:
            d = datetime.strptime(text, f)
            return d.date().isoformat()
        except Exception:
            continue
    raise ValueError(f"Unrecognized date format: {text}")

def parse_value(text: str, dtype: str):
    if text is None:
        return ""
    text = str(text).strip()
    if text == "":
        return ""
    if dtype == "string":
        return text
    if dtype == "int":
        try:
            return int(float(text)) if "." in text else int(text)
        except Exception:
            raise ValueError(f"Cannot convert '{text}' to int.")
    if dtype == "float":
        try:
            return float(text)
        except Exception:
            raise ValueError(f"Cannot convert '{text}' to float.")
    if dtype == "date":
        return try_parse_date(text)
    if dtype == "bool":
        low = text.lower()
        if low in ("true", "t", "1", "yes", "y"):
            return True
        if low in ("false", "f", "0", "no", "n"):
            return False
        raise ValueError(f"Cannot convert '{text}' to bool. Use true/false/1/0/yes/no.")
    return text

class ScrollableFrame(ttk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.window_id = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.bind("<Configure>", self._on_frame_configure)

        # Mouse wheel bindings
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel_windows)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel_linux)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel_linux)

    def _on_frame_configure(self, event):
        canvas_width = event.width
        self.canvas.itemconfig(self.window_id, width=canvas_width)

    def _on_mousewheel_windows(self, event):
        delta = int(-1 * (event.delta / 120))
        self.canvas.yview_scroll(delta, "units")

    def _on_mousewheel_linux(self, event):
        if event.num == 4:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(1, "units")

class App:
    def __init__(self, root):
        self.root = root
        root.title("CSV Row Editor")
        root.geometry("900x650")

        main = ttk.Frame(root, padding=8)
        main.pack(fill="both", expand=True)

        self.scrollframe = ScrollableFrame(main)
        self.scrollframe.pack(fill="both", expand=True)

        header_style = ttk.Style()
        header_style.configure("Header.TLabel", font=("TkDefaultFont", 10, "bold"))

        self.entries = {}
        self.datatype_vars = {}

        label_header = ttk.Label(self.scrollframe.scrollable_frame, text="Field", style="Header.TLabel")
        label_value = ttk.Label(self.scrollframe.scrollable_frame, text="Value", style="Header.TLabel")
        label_type = ttk.Label(self.scrollframe.scrollable_frame, text="Type", style="Header.TLabel")
        label_header.grid(row=0, column=0, padx=6, pady=4, sticky="w")
        label_value.grid(row=0, column=1, padx=6, pady=4, sticky="w")
        label_type.grid(row=0, column=2, padx=6, pady=4, sticky="w")

        for i, h in enumerate(HEADERS, start=1):
            lbl = ttk.Label(self.scrollframe.scrollable_frame, text=h)
            ent = ttk.Entry(self.scrollframe.scrollable_frame, width=72)
            dtype_var = tk.StringVar(value="string")
            dtype_menu = ttk.OptionMenu(self.scrollframe.scrollable_frame, dtype_var, "string", *DATATYPES)

            lbl.grid(row=i, column=0, sticky="w", padx=6, pady=2)
            ent.grid(row=i, column=1, sticky="w", padx=6, pady=2)
            dtype_menu.grid(row=i, column=2, sticky="w", padx=6, pady=2)

            self.entries[h] = ent
            self.datatype_vars[h] = dtype_var

        bar = ttk.Frame(main, padding=(0,6,0,0))
        bar.pack(fill="x", side="bottom")

        btn_load = ttk.Button(bar, text="Load CSV", command=self.load_csv)
        btn_reset = ttk.Button(bar, text="Reset", command=self.reset_form)
        btn_save = ttk.Button(bar, text="Save to CSV", command=self.save_csv)

        btn_load.pack(side="left", padx=6)
        btn_reset.pack(side="left", padx=6)
        btn_save.pack(side="left", padx=6)

        self.status_var = tk.StringVar(value="")
        status = ttk.Label(bar, textvariable=self.status_var)
        status.pack(side="right", padx=6)

    def reset_form(self):
        for h, ent in self.entries.items():
            ent.delete(0, tk.END)
            self.datatype_vars[h].set("string")
        self.status_var.set("Form reset.")

    def gather_row(self):
        row = {}
        for h in HEADERS:
            raw = self.entries[h].get()
            dtype = self.datatype_vars[h].get()
            row[h] = (raw, dtype)
        return row

    def save_csv(self):
        rowpairs = self.gather_row()
        converted = {}
        errors = []
        for h, (raw, dtype) in rowpairs.items():
            try:
                val = parse_value(raw, dtype)
                if isinstance(val, bool):
                    val_out = "True" if val else "False"
                else:
                    val_out = val
                converted[h] = val_out
            except ValueError as e:
                errors.append(f"{h}: {e}")

        if errors:
            msg = "Errors during conversion:\n" + "\n".join(errors)
            messagebox.showerror("Conversion error", msg)
            self.status_var.set("Save failed (conversion error).")
            return

        default_name = f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", initialfile=default_name,
                                                 filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not file_path:
            self.status_var.set("Save canceled.")
            return

        try:
            with open(file_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, delimiter="\t", quotechar='"')
                writer.writerow(HEADERS)
                row_out = []
                for h in HEADERS:
                    v = converted.get(h, "")
                    row_out.append(v)
                writer.writerow(row_out)
            self.status_var.set(f"Saved: {file_path}")
            messagebox.showinfo("Saved", f"CSV saved to:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Write error", f"Failed to write CSV: {e}")
            self.status_var.set("Save failed (write error).")

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not file_path:
            self.status_var.set("Load canceled.")
            return

        try:
            with open(file_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t", quotechar='"')
                rows = list(reader)
        except Exception as e:
            messagebox.showerror("Read error", f"Failed to read CSV: {e}")
            self.status_var.set("Load failed (read error).")
            return

        if not rows:
            messagebox.showerror("Empty CSV", "CSV contains no data rows to load.")
            self.status_var.set("Load failed (empty CSV).")
            return

        if len(rows) == 1:
            self._load_row_dict(rows[0])
            self.status_var.set(f"Loaded 1 row from: {file_path}")
            return

        # Multiple rows: present a small selector to pick which row to load
        selector = tk.Toplevel(self.root)
        selector.title("Select row to load")
        selector.geometry("700x400")
        selector.transient(self.root)
        selector.grab_set()

        info = ttk.Label(selector, text=f"Multiple rows found in {file_path}. Select one to load:")
        info.pack(anchor="w", padx=8, pady=6)

        frame = ttk.Frame(selector)
        frame.pack(fill="both", expand=True, padx=8, pady=(0,8))

        sb = ttk.Scrollbar(frame, orient="vertical")
        listbox = tk.Listbox(frame, yscrollcommand=sb.set, selectmode="browse")
        sb.config(command=listbox.yview)
        sb.pack(side="right", fill="y")
        listbox.pack(side="left", fill="both", expand=True)

        # prepare mapping for display: show first few header columns (id, startdate, PID if present)
        display_keys = [k for k in ("id", "startdate", "PID") if k in reader.fieldnames]  # preserves order
        if not display_keys:
            display_keys = reader.fieldnames[:3]  # fallback to first three CSV headers

        for i, r in enumerate(rows):
            parts = []
            for k in display_keys:
                v = r.get(k, "")
                parts.append(f"{k}={v}" if v is not None else f"{k}=")
            preview = "; ".join(parts)
            listbox.insert("end", f"Row {i+1}: {preview}")

        def on_load_selected(event=None):
            sel = listbox.curselection()
            if not sel:
                messagebox.showwarning("Select row", "Please select a row to load.")
                return
            idx = sel[0]
            self._load_row_dict(rows[idx])
            selector.grab_release()
            selector.destroy()
            self.status_var.set(f"Loaded row {idx+1} from: {file_path}")

        def on_double_click(event):
            on_load_selected()

        btn_frame = ttk.Frame(selector)
        btn_frame.pack(fill="x", padx=8, pady=6)
        btn_load = ttk.Button(btn_frame, text="Load selected row", command=on_load_selected)
        btn_cancel = ttk.Button(btn_frame, text="Cancel", command=lambda: (selector.grab_release(), selector.destroy()))
        btn_load.pack(side="left", padx=(0,6))
        btn_cancel.pack(side="left")

        listbox.bind("<Double-Button-1>", on_double_click)

    def _load_row_dict(self, row_dict):
        # Map CSV headers to our HEADERS case-insensitively where possible
        csv_keys = list(row_dict.keys())
        lower_map = {h.lower(): h for h in HEADERS}
        mapped = {}
        for k in csv_keys:
            if k in HEADERS:
                mapped[k] = k
            else:
                kl = k.lower()
                if kl in lower_map:
                    mapped[k] = lower_map[kl]
        # For each header in HEADERS, pick from csv using either exact key or case-insensitive match
        for h in HEADERS:
            # look for exact key in row_dict; else try case-insensitive key
            val = ""
            if h in row_dict:
                val = row_dict[h] if row_dict[h] is not None else ""
            else:
                # try find CSV key that maps to this header
                found = None
                for csv_k in row_dict.keys():
                    if csv_k.lower() == h.lower():
                        found = csv_k
                        break
                if found is not None:
                    val = row_dict.get(found, "")
            # Insert into entry (replace existing)
            ent = self.entries.get(h)
            if ent is not None:
                ent.delete(0, tk.END)
                # write as string
                ent.insert(0, "" if val is None else str(val))
        # note: datatypes are left unchanged
        return

def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()

if __name__ == "__main__":
    main()