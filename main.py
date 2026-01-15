# -*- coding: utf-8 -*-

import sys
import os

# Fix Qt platform plugin error
if hasattr(sys, 'frozen'):
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = os.path.dirname(sys.executable)
else:
    import PyQt5
    pyqt_path = os.path.dirname(PyQt5.__file__)
    plugin_path = os.path.join(pyqt_path, 'Qt5', 'plugins', 'platforms')
    if os.path.exists(plugin_path):
        os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path

import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd

from PyQt5.QtCore import Qt, QAbstractTableModel, QModelIndex, QVariant, QDate
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFrame, QLabel, QPushButton, QHBoxLayout, QVBoxLayout,
    QGridLayout, QFileDialog, QStackedWidget, QComboBox, QDateEdit, QMessageBox, QTableView,
    QScrollArea, QSizePolicy, QSpacerItem
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


def get_base_path():
    """Get the base path for resources - works for both script and frozen exe"""
    if getattr(sys, 'frozen', False):
        # Running as compiled exe
        return os.path.dirname(sys.executable)
    else:
        # Running as script
        return os.path.dirname(os.path.abspath(__file__))

BASE_PATH = get_base_path()


LOG_PATH = os.path.join(BASE_PATH, "app.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("cw1_dashboard")


DARK_QSS = """
QMainWindow { background: #0b1220; }
QWidget { color: #e5e7eb; font-family: Inter, Segoe UI, Arial; font-size: 12px; }
QFrame#sidebar { background: #0f172a; border-right: 1px solid #1f2937; }
QLabel#appTitle { font-size: 18px; font-weight: 700; color: #f9fafb; }
QLabel#muted { color: #9ca3af; }

QFrame#card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 14px;
}
QLabel#kpiValue { font-size: 22px; font-weight: 800; color: #f9fafb; }
QLabel#kpiLabel { color: #9ca3af; }

QFrame#topbar {
    background: #0b1220;
    border-bottom: 1px solid #1f2937;
}
QPushButton {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 10px;
    padding: 8px 10px;
}
QPushButton:hover { border-color: #374151; }
QPushButton#navBtn {
    background: transparent;
    border: none;
    text-align: left;
    padding: 10px 12px;
    border-radius: 10px;
}
QPushButton#navBtn:hover { background: #111827; }
QPushButton#navBtn[active="true"] { background: #111827; border: 1px solid #1f2937; }

QComboBox, QDateEdit {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 10px;
    padding: 6px 10px;
}
QTableView {
    background: #0f172a;
    gridline-color: #1f2937;
    border: 1px solid #1f2937;
    border-radius: 12px;
}
QHeaderView::section {
    background: #111827;
    color: #e5e7eb;
    border: none;
    padding: 6px 8px;
}
"""

LIGHT_QSS = """
QMainWindow { background: #f8fafc; }
QWidget { color: #0f172a; font-family: Inter, Segoe UI, Arial; font-size: 12px; }
QFrame#sidebar { background: #ffffff; border-right: 1px solid #e5e7eb; }
QLabel#appTitle { font-size: 18px; font-weight: 700; color: #0f172a; }
QLabel#muted { color: #64748b; }

QFrame#card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
}
QLabel#kpiValue { font-size: 22px; font-weight: 800; color: #0f172a; }
QLabel#kpiLabel { color: #64748b; }

QFrame#topbar {
    background: #f8fafc;
    border-bottom: 1px solid #e5e7eb;
}
QPushButton {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 8px 10px;
}
QPushButton:hover { border-color: #cbd5e1; }
QPushButton#navBtn {
    background: transparent;
    border: none;
    text-align: left;
    padding: 10px 12px;
    border-radius: 10px;
}
QPushButton#navBtn:hover { background: #f1f5f9; }
QPushButton#navBtn[active="true"] { background: #f1f5f9; border: 1px solid #e5e7eb; }

QComboBox, QDateEdit {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 6px 10px;
}
QTableView {
    background: #ffffff;
    gridline-color: #e5e7eb;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
}
QHeaderView::section {
    background: #f1f5f9;
    color: #0f172a;
    border: none;
    padding: 6px 8px;
}
"""


@dataclass
class Filters:
    brand: str = "All"
    country: str = "All"
    sentiment: str = "All"
    date_from: Optional[pd.Timestamp] = None
    date_to: Optional[pd.Timestamp] = None


class DataRepository:
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.legend: Optional[pd.DataFrame] = None
        self.path: Optional[str] = None

    def load_excel(self, path: str) -> None:
        logger.info("Loading Excel: %s", path)
        self.path = path

        legend = pd.read_excel(path, sheet_name="legend")
        raw = pd.read_excel(path, sheet_name="raw data")

        legend_vars = set(legend["VARIABLE"].dropna().astype(str).tolist())
        keep_cols = [c for c in raw.columns if str(c) in legend_vars]
        df = raw[keep_cols].copy()

        df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
        for col in ["verified_purchase"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(0, 1)

        df = df.dropna(subset=["review_id"])
        self.df = df
        self.legend = legend
        logger.info("Loaded rows=%s cols=%s", df.shape[0], df.shape[1])

    def get_filtered(self, f: Filters) -> pd.DataFrame:
        if self.df is None:
            return pd.DataFrame()

        df = self.df

        if f.brand != "All":
            df = df[df["brand"] == f.brand]
        if f.country != "All":
            df = df[df["country"] == f.country]
        if f.sentiment != "All":
            df = df[df["sentiment"] == f.sentiment]
        if f.date_from is not None:
            df = df[df["review_date"] >= f.date_from]
        if f.date_to is not None:
            df = df[df["review_date"] <= f.date_to]

        return df

class PandasTableModel(QAbstractTableModel):
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df

    def set_df(self, df: pd.DataFrame):
        self.beginResetModel()
        self._df = df
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if self._df is None else len(self._df)

    def columnCount(self, parent=QModelIndex()):
        return 0 if self._df is None else self._df.shape[1]

    def data(self, index: QModelIndex, role=Qt.DisplayRole):
        if not index.isValid() or self._df is None:
            return QVariant()

        if role == Qt.DisplayRole:
            val = self._df.iat[index.row(), index.column()]
            if pd.isna(val):
                return ""
            if isinstance(val, (float, np.floating)):
                return f"{val:,.2f}"
            return str(val)

        return QVariant()

    def headerData(self, section: int, orientation: Qt.Orientation, role=Qt.DisplayRole):
        if self._df is None or role != Qt.DisplayRole:
            return QVariant()
        if orientation == Qt.Horizontal:
            return str(self._df.columns[section])
        return str(section + 1)


class MplCanvas(FigureCanvas):
    def __init__(self, width=5, height=3, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)


class KpiCard(QFrame):
    def __init__(self, label: str, value: str):
        super().__init__()
        self.setObjectName("card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(6)

        self.lbl_value = QLabel(value)
        self.lbl_value.setObjectName("kpiValue")

        self.lbl_label = QLabel(label)
        self.lbl_label.setObjectName("kpiLabel")

        layout.addWidget(self.lbl_value)
        layout.addWidget(self.lbl_label)

    def set_value(self, value: str):
        self.lbl_value.setText(value)


class OverviewPage(QWidget):
    def __init__(self):
        super().__init__()

        self.filters_ui: Dict[str, Any] = {}
        self.kpis: Dict[str, KpiCard] = {}
        self.charts: Dict[str, MplCanvas] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        # Filter bar
        filter_frame = QFrame()
        filter_frame.setObjectName("card")
        fb = QHBoxLayout(filter_frame)
        fb.setContentsMargins(14, 10, 14, 10)
        fb.setSpacing(10)

        fb.addWidget(self._make_label("Brand"))
        brand = QComboBox()
        fb.addWidget(brand)

        fb.addWidget(self._make_label("Country"))
        country = QComboBox()
        fb.addWidget(country)

        fb.addWidget(self._make_label("Sentiment"))
        sentiment = QComboBox()
        fb.addWidget(sentiment)

        fb.addWidget(self._make_label("From"))
        date_from = QDateEdit()
        date_from.setCalendarPopup(True)
        fb.addWidget(date_from)

        fb.addWidget(self._make_label("To"))
        date_to = QDateEdit()
        date_to.setCalendarPopup(True)
        fb.addWidget(date_to)

        apply_btn = QPushButton("Apply")
        fb.addWidget(apply_btn)

        fb.addItem(QSpacerItem(10, 10, QSizePolicy.Expanding, QSizePolicy.Minimum))

        export_btn = QPushButton("Export view (PNG)")
        fb.addWidget(export_btn)

        self.filters_ui = {
            "brand": brand,
            "country": country,
            "sentiment": sentiment,
            "date_from": date_from,
            "date_to": date_to,
            "apply_btn": apply_btn,
            "export_btn": export_btn,
        }

        root.addWidget(filter_frame)


        kpi_grid = QGridLayout()
        kpi_grid.setHorizontalSpacing(12)
        kpi_grid.setVerticalSpacing(12)

        self.kpis["reviews"] = KpiCard("Total reviews", "—")
        self.kpis["avg_rating"] = KpiCard("Average rating", "—")
        self.kpis["verified"] = KpiCard("Verified purchases", "—")
        self.kpis["avg_price"] = KpiCard("Average price (USD)", "—")
        self.kpis["pos_share"] = KpiCard("Positive share", "—")
        self.kpis["avg_helpful"] = KpiCard("Avg helpful votes", "—")

        cards = list(self.kpis.values())
        for i, card in enumerate(cards):
            r = i // 3
            c = i % 3
            kpi_grid.addWidget(card, r, c)

        root.addLayout(kpi_grid)

        # Charts area (scrollable)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        charts_container = QWidget()
        charts_layout = QGridLayout(charts_container)
        charts_layout.setContentsMargins(0, 0, 0, 0)
        charts_layout.setHorizontalSpacing(12)
        charts_layout.setVerticalSpacing(12)

        self.charts["trend"] = self._chart_card("Reviews over time (monthly)")
        self.charts["sentiment"] = self._chart_card("Sentiment distribution")
        self.charts["brands"] = self._chart_card("Top brands by review count")
        self.charts["aspects"] = self._chart_card("Average aspect ratings")

        charts_layout.addWidget(self._wrap_canvas(self.charts["trend"]), 0, 0)
        charts_layout.addWidget(self._wrap_canvas(self.charts["sentiment"]), 0, 1)
        charts_layout.addWidget(self._wrap_canvas(self.charts["brands"]), 1, 0)
        charts_layout.addWidget(self._wrap_canvas(self.charts["aspects"]), 1, 1)

        scroll.setWidget(charts_container)
        root.addWidget(scroll, 1)

    @staticmethod
    def _make_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("muted")
        return lbl

    @staticmethod
    def _wrap_canvas(canvas: MplCanvas) -> QFrame:
        frame = QFrame()
        frame.setObjectName("card")
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(12, 10, 12, 12)
        lay.setSpacing(8)
        title = QLabel(canvas.property("title") or "")
        title.setStyleSheet("font-weight: 700;")
        lay.addWidget(title)
        lay.addWidget(canvas)
        return frame

    @staticmethod
    def _chart_card(title: str) -> MplCanvas:
        c = MplCanvas(width=5, height=3, dpi=100)
        c.setProperty("title", title)
        return c


class LegendPage(QWidget):
    def __init__(self):
        super().__init__()
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        title = QLabel("Data dictionary (legend)")
        title.setStyleSheet("font-size: 16px; font-weight: 800;")
        root.addWidget(title)

        self.table = QTableView()
        self.table.setAlternatingRowColors(True)
        root.addWidget(self.table, 1)

        self.model = PandasTableModel(pd.DataFrame(columns=["VARIABLE", "DESCRIPTION"]))
        self.table.setModel(self.model)

    def set_legend(self, legend_df: pd.DataFrame):
        cols = list(legend_df.columns)
        var_col = cols[0]
        desc_col = cols[-1]
        cleaned = legend_df[[var_col, desc_col]].copy()
        cleaned.columns = ["VARIABLE", "DESCRIPTION"]
        self.model.set_df(cleaned)


class DataTablePage(QWidget):
    def __init__(self):
        super().__init__()
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        title = QLabel("Filtered data")
        title.setStyleSheet("font-size: 16px; font-weight: 800;")
        root.addWidget(title)

        self.table = QTableView()
        self.table.setSortingEnabled(False)
        self.table.setAlternatingRowColors(True)

        self.model = PandasTableModel(pd.DataFrame())
        self.table.setModel(self.model)
        root.addWidget(self.table, 1)

    def set_df(self, df: pd.DataFrame):
        self.model.set_df(df)


class MainWindow(QMainWindow):
    def __init__(self, default_path: Optional[str] = None):
        super().__init__()
        self.setWindowTitle("CW1 Dashboard (PyQt5)")
        self.resize(1300, 820)

        self.repo = DataRepository()
        self.filters = Filters()
        self.is_dark = False

        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self.sidebar = QFrame()
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setFixedWidth(240)
        sb = QVBoxLayout(self.sidebar)
        sb.setContentsMargins(16, 16, 16, 16)
        sb.setSpacing(10)

        app_title = QLabel("CW1 Dashboard")
        app_title.setObjectName("appTitle")
        sb.addWidget(app_title)

        subtitle = QLabel("Excel → Desktop app\nFilters + Charts + Export")
        subtitle.setObjectName("muted")
        sb.addWidget(subtitle)

        sb.addSpacing(10)

        self.btn_overview = self._nav_button("Overview")
        self.btn_data = self._nav_button("Data table")
        self.btn_legend = self._nav_button("Legend")
        self.btn_open = QPushButton("Open Excel…")

        sb.addWidget(self.btn_overview)
        sb.addWidget(self.btn_data)
        sb.addWidget(self.btn_legend)
        sb.addSpacing(10)
        sb.addWidget(self.btn_open)
        sb.addItem(QSpacerItem(10, 10, QSizePolicy.Minimum, QSizePolicy.Expanding))

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        self.topbar = QFrame()
        self.topbar.setObjectName("topbar")
        tb = QHBoxLayout(self.topbar)
        tb.setContentsMargins(16, 12, 16, 12)
        tb.setSpacing(10)

        self.lbl_file = QLabel("No file loaded")
        self.lbl_file.setObjectName("muted")

        self.btn_refresh = QPushButton("Refresh")
        self.btn_theme = QPushButton("Light mode")

        tb.addWidget(self.lbl_file)
        tb.addItem(QSpacerItem(10, 10, QSizePolicy.Expanding, QSizePolicy.Minimum))
        tb.addWidget(self.btn_refresh)
        tb.addWidget(self.btn_theme)

        # Pages
        self.pages = QStackedWidget()
        self.page_overview = OverviewPage()
        self.page_data = DataTablePage()
        self.page_legend = LegendPage()

        self.pages.addWidget(self.page_overview)
        self.pages.addWidget(self.page_data)
        self.pages.addWidget(self.page_legend)

        content_layout.addWidget(self.topbar)
        content_layout.addWidget(self.pages, 1)

        root_layout.addWidget(self.sidebar)
        root_layout.addWidget(content, 1)

        self.setCentralWidget(root)

        # Wiring
        self.btn_overview.clicked.connect(lambda: self._set_page(0))
        self.btn_data.clicked.connect(lambda: self._set_page(1))
        self.btn_legend.clicked.connect(lambda: self._set_page(2))

        self.btn_open.clicked.connect(self.open_file)
        self.btn_refresh.clicked.connect(self.refresh)
        self.btn_theme.clicked.connect(self.toggle_theme)

        self.page_overview.filters_ui["apply_btn"].clicked.connect(self.apply_filters)
        self.page_overview.filters_ui["export_btn"].clicked.connect(self.export_view)


        self.apply_theme()

        if default_path and os.path.exists(default_path):
            self.load_dataset(default_path)

        self._set_page(0)

    def _nav_button(self, text: str) -> QPushButton:
        btn = QPushButton(text)
        btn.setObjectName("navBtn")
        btn.setProperty("active", "false")
        btn.setCursor(Qt.PointingHandCursor)
        return btn

    def _set_page(self, idx: int):
        self.pages.setCurrentIndex(idx)
        for b in [self.btn_overview, self.btn_data, self.btn_legend]:
            b.setProperty("active", "false")
            b.style().unpolish(b)
            b.style().polish(b)

        active = [self.btn_overview, self.btn_data, self.btn_legend][idx]
        active.setProperty("active", "true")
        active.style().unpolish(active)
        active.style().polish(active)

    def apply_theme(self):
        qss = DARK_QSS if self.is_dark else LIGHT_QSS
        self.setStyleSheet(qss)
        self.btn_theme.setText("Light mode" if self.is_dark else "Dark mode")

    def toggle_theme(self):
        self.is_dark = not self.is_dark
        self.apply_theme()
        self.refresh()

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open dataset", "", "Excel (*.xlsx *.xls)")
        if path:
            self.load_dataset(path)

    def load_dataset(self, path: str):
        try:
            self.repo.load_excel(path)
        except Exception as e:
            logger.exception("Failed to load dataset")
            QMessageBox.critical(self, "Load error", f"Could not load Excel:\n{e}")
            return

        self.lbl_file.setText(os.path.basename(path))
        self._populate_filters()
        self.refresh()

        if self.repo.legend is not None:
            self.page_legend.set_legend(self.repo.legend)

    def _populate_filters(self):
        df = self.repo.df
        if df is None or df.empty:
            return

        brand_cb: QComboBox = self.page_overview.filters_ui["brand"]
        country_cb: QComboBox = self.page_overview.filters_ui["country"]
        sentiment_cb: QComboBox = self.page_overview.filters_ui["sentiment"]
        date_from: QDateEdit = self.page_overview.filters_ui["date_from"]
        date_to: QDateEdit = self.page_overview.filters_ui["date_to"]

        def fill(cb: QComboBox, values: List[str]):
            cb.blockSignals(True)
            cb.clear()
            cb.addItem("All")
            for v in values:
                cb.addItem(str(v))
            cb.blockSignals(False)

        fill(brand_cb, sorted(df["brand"].dropna().unique().tolist()))
        fill(country_cb, sorted(df["country"].dropna().unique().tolist()))
        fill(sentiment_cb, sorted(df["sentiment"].dropna().unique().tolist()))

        # Date range
        dmin = df["review_date"].min()
        dmax = df["review_date"].max()
        if pd.notna(dmin) and pd.notna(dmax):
            date_from.setDate(QDate(dmin.year, dmin.month, dmin.day))
            date_to.setDate(QDate(dmax.year, dmax.month, dmax.day))

    def apply_filters(self):
        if self.repo.df is None:
            return

        ui = self.page_overview.filters_ui
        brand = ui["brand"].currentText()
        country = ui["country"].currentText()
        sentiment = ui["sentiment"].currentText()

        qdf: QDateEdit = ui["date_from"]
        qdt: QDateEdit = ui["date_to"]

        date_from = pd.Timestamp(qdf.date().toPyDate())
        date_to = pd.Timestamp(qdt.date().toPyDate())

        self.filters = Filters(
            brand=brand,
            country=country,
            sentiment=sentiment,
            date_from=date_from,
            date_to=date_to,
        )
        self.refresh()

    def refresh(self):
        df = self.repo.get_filtered(self.filters)
        if df.empty:
            return

        self._update_kpis(df)
        self._update_charts(df)
        self.page_data.set_df(df)

    def _update_kpis(self, df: pd.DataFrame):
        total = len(df)
        avg_rating = df["rating"].mean()
        verified = df["verified_purchase"].mean() if "verified_purchase" in df.columns else np.nan
        avg_price = df["price_usd"].mean() if "price_usd" in df.columns else np.nan
        pos_share = (df["sentiment"].eq("Positive").mean()) if "sentiment" in df.columns else np.nan
        avg_helpful = df["helpful_votes"].mean() if "helpful_votes" in df.columns else np.nan

        self.page_overview.kpis["reviews"].set_value(f"{total:,}")
        self.page_overview.kpis["avg_rating"].set_value(f"{avg_rating:.2f}")
        self.page_overview.kpis["verified"].set_value(f"{verified*100:.1f}%")
        self.page_overview.kpis["avg_price"].set_value(f"${avg_price:,.0f}")
        self.page_overview.kpis["pos_share"].set_value(f"{pos_share*100:.1f}%")
        self.page_overview.kpis["avg_helpful"].set_value(f"{avg_helpful:.2f}")

    def _style_axes(self, ax):
        if self.is_dark:
            ax.set_facecolor("#111827")
            ax.figure.set_facecolor("#111827")
            ax.tick_params(colors="#e5e7eb")
            ax.xaxis.label.set_color("#e5e7eb")
            ax.yaxis.label.set_color("#e5e7eb")
            ax.title.set_color("#f9fafb")
            for spine in ax.spines.values():
                spine.set_color("#374151")
        else:
            ax.set_facecolor("#ffffff")
            ax.figure.set_facecolor("#ffffff")
            ax.tick_params(colors="#0f172a")
            ax.xaxis.label.set_color("#0f172a")
            ax.yaxis.label.set_color("#0f172a")
            ax.title.set_color("#0f172a")
            for spine in ax.spines.values():
                spine.set_color("#cbd5e1")

    def _update_charts(self, df: pd.DataFrame):
        c = self.page_overview.charts["trend"]
        ax = c.ax
        ax.clear()

        monthly = (
            df.assign(month=df["review_date"].dt.to_period("M").dt.to_timestamp())
              .groupby("month")["review_id"].count()
        )
        ax.plot(monthly.index, monthly.values)
        ax.set_title("Reviews over time")
        ax.set_xlabel("Month")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=30)
        self._style_axes(ax)
        c.draw()
        
        c = self.page_overview.charts["sentiment"]
        ax = c.ax
        ax.clear()

        s = df["sentiment"].value_counts()
        ax.bar(s.index.astype(str), s.values)
        ax.set_title("Sentiment distribution")
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Reviews")
        self._style_axes(ax)
        c.draw()
        c = self.page_overview.charts["brands"]
        ax = c.ax
        ax.clear()

        b = df["brand"].value_counts().head(10).sort_values()
        ax.barh(b.index.astype(str), b.values)
        ax.set_title("Top brands by reviews")
        ax.set_xlabel("Reviews")
        self._style_axes(ax)
        c.draw()
        c = self.page_overview.charts["aspects"]
        ax = c.ax
        ax.clear()

        aspect_cols = [x for x in [
            "battery_life_rating", "camera_rating", "performance_rating", "design_rating", "display_rating"
        ] if x in df.columns]

        if aspect_cols:
            means = df[aspect_cols].mean().sort_values()
            ax.barh([x.replace("_rating", "").replace("_", " ").title() for x in means.index], means.values)
            ax.set_title("Average aspect ratings")
            ax.set_xlabel("Avg score")
            ax.set_xlim(0, 5)
        else:
            ax.text(0.5, 0.5, "No aspect columns found", ha="center", va="center")
            ax.set_axis_off()

        self._style_axes(ax)
        c.draw()

    def export_view(self):
        if self.pages.currentWidget() is not self.page_overview:
            self._set_page(0)

        path, _ = QFileDialog.getSaveFileName(self, "Save PNG", "dashboard.png", "PNG (*.png)")
        if not path:
            return

        pix = self.page_overview.grab()
        ok = pix.save(path, "PNG")
        if ok:
            QMessageBox.information(self, "Saved", f"Exported:\n{path}")
        else:
            QMessageBox.warning(self, "Error", "Could not save image.")


def main():
    app = QApplication(sys.argv)

    default_path = os.path.join(BASE_PATH, "CW1_data.xlsx")
    win = MainWindow(default_path=default_path if os.path.exists(default_path) else None)
    win.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
