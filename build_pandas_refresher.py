"""Build the learner and solutions notebooks for the Pandas learning-by-doing refresher."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LEARNER = ROOT / "pandas_learning_by_doing.ipynb"
SOLUTIONS = ROOT / "pandas_learning_by_doing_solutions.ipynb"


learner_cells: list[dict] = []
solution_cells: list[dict] = []
cell_counter = 0
exercise_counter = 0


def _cell(kind: str, source: str) -> dict:
    global cell_counter
    cell_counter += 1
    base = {
        "cell_type": kind,
        "id": f"cell-{cell_counter:03d}",
        "metadata": {},
        "source": source.strip("\n").splitlines(keepends=True),
    }
    if kind == "code":
        base.update({"execution_count": None, "outputs": []})
    return base


def both_md(source: str) -> None:
    learner_cells.append(_cell("markdown", source))
    solution_cells.append(_cell("markdown", source))


def both_code(source: str) -> None:
    learner_cells.append(_cell("code", source))
    solution_cells.append(_cell("code", source))


def exercise(
    title: str,
    prompt: str,
    starter: str,
    solution: str,
    *,
    hint: str = "",
    expected: str = "",
    stretch: bool = False,
) -> None:
    global exercise_counter
    exercise_counter += 1
    badge = " · 🧠 Stretch" if stretch else ""
    extras = []
    if expected:
        extras.append(f"**Target:** {expected}")
    if hint:
        extras.append(f"<details><summary>Hint</summary>{hint}</details>")
    extra_text = "\n\n".join(extras)
    heading = f"### Exercise {exercise_counter}: {title}{badge}\n\n{prompt}"
    if extra_text:
        heading += "\n\n" + extra_text
    learner_cells.append(_cell("markdown", heading))
    learner_cells.append(_cell("code", starter))
    solution_cells.append(_cell("markdown", heading + "\n\n**Worked solution:**"))
    solution_cells.append(_cell("code", solution))


both_md(
    r"""
# Pandas: close the gap today

This is a **learning-by-doing** refresher, not a tour of methods. You will work with a small but realistic commerce dataset and repeatedly answer questions an analyst actually gets asked.

## How to use this notebook

1. Run the setup cells once.
2. For every exercise, **predict the shape and columns first**.
3. Write code from memory. Use the hint only after 3–5 honest minutes.
4. Check the companion solutions notebook only after you have a result.
5. When your answer differs, explain *why* in a new Markdown cell.

### Suggested one-day route

| Block | Sections | Time |
|---|---|---:|
| Warm-up | 0–2 | 75 min |
| Data wrangling | 3–4 | 90 min |
| Analysis | 5–6 | 90 min |
| Advanced patterns | 7–9 | 90 min |
| Capstone | 10 | 2–3 hr |

Exercises marked **Stretch** are optional today. The core track is enough to rebuild fluency.

> **Rule:** do not mutate the source tables (`customers`, `products`, `orders`, `order_items`, `returns`, `events`). Create a copy when an exercise needs mutation.
"""
)

both_code(
    r"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    from IPython.display import display
except ImportError:  # Keeps the notebook runnable in plain Python too.
    display = print

pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

print("pandas", pd.__version__)
"""
)

both_md(
    r"""
## 0. Build the practice dataset

Everything is deterministic. Re-running this cell restores the original tables.

**Data model**

- `customers`: one row per customer
- `products`: one row per product
- `orders`: one row per order
- `order_items`: one or more rows per order
- `returns`: a subset of returned order items
- `events`: customer browsing events
"""
)

both_code(
    r"""
rng = np.random.default_rng(42)

# Customers
n_customers = 60
customer_ids = np.arange(1001, 1001 + n_customers)
cities = np.array(["Riyadh", "Jeddah", "Dammam", "Makkah", "Medina", None], dtype=object)
segments = np.array(["Consumer", "Corporate", "Student"])
first_names = np.array(["Lina", "Omar", "Sara", "Noura", "Faisal", "Maya", "Ali", "Reem"])
last_names = np.array(["Khan", "Saleh", "Ahmed", "Noor", "Hassan", "Yousef"])

customers = pd.DataFrame({
    "customer_id": customer_ids,
    "first_name": rng.choice(first_names, n_customers),
    "last_name": rng.choice(last_names, n_customers),
    "segment": rng.choice(segments, n_customers, p=[0.55, 0.25, 0.20]),
    "city": rng.choice(cities, n_customers, p=[0.30, 0.25, 0.16, 0.12, 0.10, 0.07]),
    "signup_date": pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 730, n_customers), unit="D"),
    "birth_year": rng.choice([1985, 1990, 1995, 2000, 2003, np.nan], n_customers),
    "is_active": rng.choice([True, False], n_customers, p=[0.88, 0.12]),
})
customers["email"] = (
    " " + customers["first_name"].str.upper() + "." + customers["last_name"].str.lower()
    + customers["customer_id"].astype(str) + "@Example.COM "
)
customers["segment"] = customers["segment"].astype("category")

# Products
n_products = 24
product_ids = np.arange(201, 201 + n_products)
product_categories = np.array(["Electronics", "Home", "Office", "Fitness"])
category_for_product = np.repeat(product_categories, 6)
unit_cost = rng.integers(20, 450, n_products).astype(float)
markup = rng.uniform(1.25, 2.10, n_products)
products = pd.DataFrame({
    "product_id": product_ids,
    "product_name": [f"{cat[:3].upper()}-{i:02d}" for i, cat in enumerate(category_for_product, 1)],
    "category": category_for_product,
    "unit_cost": unit_cost,
    "list_price": np.round(unit_cost * markup, 2),
    "supplier": rng.choice(["Northstar", "Gulf Goods", "Vertex"], n_products),
})
products["category"] = products["category"].astype("category")

# Orders
n_orders = 420
order_ids = np.arange(50001, 50001 + n_orders)
order_ts = (
    pd.Timestamp("2025-01-01")
    + pd.to_timedelta(rng.integers(0, 181, n_orders), unit="D")
    + pd.to_timedelta(rng.integers(8, 23, n_orders), unit="h")
    + pd.to_timedelta(rng.integers(0, 60, n_orders), unit="m")
)
orders = pd.DataFrame({
    "order_id": order_ids,
    "customer_id": rng.choice(customer_ids, n_orders),
    "order_ts": order_ts,
    "channel": rng.choice(["Web", "App", "Store"], n_orders, p=[0.45, 0.35, 0.20]),
    "status": rng.choice(["completed", "cancelled", "pending"], n_orders, p=[0.86, 0.08, 0.06]),
    "shipping_fee": rng.choice([0.0, 15.0, 25.0], n_orders, p=[0.30, 0.48, 0.22]),
    "coupon_code": rng.choice([None, "WELCOME10", "SAVE20", "VIP"], n_orders, p=[0.65, 0.18, 0.12, 0.05]),
})

# Order items
item_rows = []
item_id = 1
product_lookup = products.set_index("product_id")["list_price"]
for order_id in order_ids:
    for product_id in rng.choice(product_ids, size=int(rng.integers(1, 6)), replace=False):
        list_price = product_lookup.loc[product_id]
        item_rows.append({
            "item_id": item_id,
            "order_id": order_id,
            "product_id": product_id,
            "quantity": int(rng.integers(1, 5)),
            "unit_price": round(float(list_price * rng.choice([0.95, 1.00, 1.05])), 2),
            "discount_pct": float(rng.choice([0, 0.05, 0.10, 0.20], p=[0.52, 0.20, 0.20, 0.08])),
        })
        item_id += 1
order_items = pd.DataFrame(item_rows)

# Returns (each item can appear at most once)
returned = order_items.sample(55, random_state=42)[["item_id", "order_id"]].copy()
returned["return_date"] = (
    returned["order_id"].map(orders.set_index("order_id")["order_ts"]).dt.normalize()
    + pd.to_timedelta(rng.integers(2, 31, len(returned)), unit="D")
)
returned["return_reason"] = rng.choice(
    ["Damaged", "Not as described", "Changed mind", "Wrong item"], len(returned)
)
returns = returned.reset_index(drop=True)

# Web/app events
n_events = 1000
events = pd.DataFrame({
    "event_id": np.arange(1, n_events + 1),
    "customer_id": rng.choice(customer_ids, n_events),
    "event_ts": (
        pd.Timestamp("2025-01-01")
        + pd.to_timedelta(rng.integers(0, 181, n_events), unit="D")
        + pd.to_timedelta(rng.integers(0, 24 * 60, n_events), unit="m")
    ),
    "event_type": rng.choice(["view", "search", "add_to_cart", "purchase"], n_events, p=[0.55, 0.20, 0.17, 0.08]),
    "device": rng.choice(["mobile", "desktop", "tablet"], n_events, p=[0.64, 0.29, 0.07]),
})

tables = {"customers": customers, "products": products, "orders": orders,
          "order_items": order_items, "returns": returns, "events": events}
pd.DataFrame({name: [df.shape[0], df.shape[1]] for name, df in tables.items()}, index=["rows", "columns"]).T
"""
)

both_code(
    r"""
def audit(df, name="DataFrame"):
    '''Compact inspection helper used throughout the notebook.'''
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "missing_pct": df.isna().mean().mul(100).round(1),
        "unique": df.nunique(dropna=False),
    }).rename_axis(name)


def check_unique(df, keys):
    '''Raise early if keys that should be unique are not unique.'''
    if df.duplicated(keys).any():
        raise ValueError(f"Expected unique key(s): {keys}")
    return True


check_unique(customers, ["customer_id"])
check_unique(products, ["product_id"])
check_unique(orders, ["order_id"])
print("Setup complete — source tables are ready.")
"""
)

both_md(
    r"""
## 1. Mental model: Series, DataFrame, index, dtype

A `Series` is a labeled one-dimensional array. A `DataFrame` is a collection of aligned Series. The index is a label system—not automatically a row number. Most pandas surprises come from **alignment**, **dtype**, or **grain** (what one row represents).
"""
)

exercise(
    "Construct and align Series",
    "Create two Series with indexes `a, b, c` and `b, c, d`, add them, then add them again with missing values filled as zero. Observe label alignment.",
    """# YOUR CODE HERE\ns1 = ...\ns2 = ...\naligned_sum = ...\nfilled_sum = ...\naligned_sum, filled_sum""",
    """s1 = pd.Series([10, 20, 30], index=["a", "b", "c"], name="left")\ns2 = pd.Series([1, 2, 3], index=["b", "c", "d"], name="right")\naligned_sum = s1 + s2\nfilled_sum = s1.add(s2, fill_value=0)\naligned_sum, filled_sum""",
    hint="Use `Series.add(..., fill_value=0)` for the second result.",
    expected="The first result has `NaN` at `a` and `d`; the second has no missing values.",
)

exercise(
    "Create a tidy DataFrame",
    "Create `mini` with columns `product`, `units`, and `price` from three equal-length Python lists. Add a vectorized `revenue` column.",
    """# YOUR CODE HERE\nmini = ...\nmini[\"revenue\"] = ...\nmini""",
    """mini = pd.DataFrame({\n    "product": ["A", "B", "C"],\n    "units": [3, 5, 2],\n    "price": [10.0, 7.5, 20.0],\n})\nmini["revenue"] = mini["units"] * mini["price"]\nmini""",
    hint="Pass a dict of lists to `pd.DataFrame`, then multiply two Series.",
)

exercise(
    "Audit before analysis",
    "Inspect `customers` with `head`, `shape`, and the provided `audit()` helper. Then write one sentence in a Markdown cell: which columns need cleaning, and why?",
    """# YOUR CODE HERE\n...""",
    """display(customers.head())\nprint(customers.shape)\ndisplay(audit(customers, "customers"))\n# Likely cleaning targets: email (spaces/case), city and birth_year (missing values).""",
    hint="Try `display(customers.head())`, `customers.shape`, and `audit(customers)`.",
)

both_md(
    r"""
## 2. Selecting, filtering, sorting, and assigning

Keep these distinctions sharp:

- `df["col"]` → Series; `df[["col"]]` → DataFrame
- `.loc[row_labels, column_labels]` → labels / boolean masks
- `.iloc[row_positions, column_positions]` → integer positions
- use `&`, `|`, and `~` for elementwise conditions; parenthesize each condition
- use `.loc[mask, "column"] = value` for conditional assignment
"""
)

exercise(
    "Series versus one-column DataFrame",
    "Select `email` from `customers` once as a Series and once as a DataFrame. Store their dimensions (`ndim`) in `selection_dims`.",
    """# YOUR CODE HERE\nemail_series = ...\nemail_frame = ...\nselection_dims = ...\nselection_dims""",
    """email_series = customers["email"]\nemail_frame = customers[["email"]]\nselection_dims = (email_series.ndim, email_frame.ndim)\nselection_dims""",
    expected="`(1, 2)`",
)

exercise(
    "Label versus position",
    "Create `customer_view` indexed by `customer_id`. Select customer IDs 1005 through 1008 with `.loc`, then select the first four rows and first three columns with `.iloc`.",
    """# YOUR CODE HERE\ncustomer_view = ...\nby_label = ...\nby_position = ...\ndisplay(by_label)\ndisplay(by_position)""",
    """customer_view = customers.set_index("customer_id")\nby_label = customer_view.loc[1005:1008]\nby_position = customer_view.iloc[:4, :3]\ndisplay(by_label)\ndisplay(by_position)""",
    hint="Remember that label slices are inclusive; position slices exclude the stop.",
)

exercise(
    "Filter with multiple conditions",
    "Find active Corporate customers in Riyadh or Jeddah. Keep only `customer_id`, `first_name`, `city`, and `segment`.",
    """# YOUR CODE HERE\nmask = ...\ntarget_customers = ...\ntarget_customers.head()""",
    """mask = (\n    customers["is_active"]\n    & customers["segment"].eq("Corporate")\n    & customers["city"].isin(["Riyadh", "Jeddah"])\n)\ntarget_customers = customers.loc[mask, ["customer_id", "first_name", "city", "segment"]]\ntarget_customers.head()""",
    hint="Combine `.eq`, `.isin`, and `&`; wrap conditions in parentheses.",
)

exercise(
    "Filter with query",
    "Use `query` to select Web orders that are not cancelled and have a shipping fee of at least 15. Sort newest first.",
    """# YOUR CODE HERE\nweb_orders = ...\nweb_orders.head()""",
    """web_orders = (\n    orders.query("channel == 'Web' and status != 'cancelled' and shipping_fee >= 15")\n          .sort_values("order_ts", ascending=False)\n)\nweb_orders.head()""",
    hint="Inside `query`, column names can be referenced directly.",
)

exercise(
    "Safe conditional assignment",
    "On a copy named `orders_labeled`, create `shipping_tier`: `free` when the fee is zero, `standard` at 15, and `express` at 25. Use `.loc` assignments.",
    """# YOUR CODE HERE\norders_labeled = orders.copy()\norders_labeled[\"shipping_tier\"] = ...\n# conditional assignments here\norders_labeled[\"shipping_tier\"].value_counts(dropna=False)""",
    """orders_labeled = orders.copy()\norders_labeled["shipping_tier"] = pd.NA\norders_labeled.loc[orders_labeled["shipping_fee"].eq(0), "shipping_tier"] = "free"\norders_labeled.loc[orders_labeled["shipping_fee"].eq(15), "shipping_tier"] = "standard"\norders_labeled.loc[orders_labeled["shipping_fee"].eq(25), "shipping_tier"] = "express"\norders_labeled["shipping_tier"].value_counts(dropna=False)""",
    hint="Initialize the column, then assign each condition with `df.loc[mask, col] = value`.",
)

exercise(
    "Top-N without sorting everything",
    "Return the five products with the largest absolute markup (`list_price - unit_cost`). Include a `markup` column.",
    """# YOUR CODE HERE\nproduct_margin = ...\ntop_markup = ...\ntop_markup""",
    """product_margin = products.assign(markup=products["list_price"] - products["unit_cost"])\ntop_markup = product_margin.nlargest(5, "markup")\ntop_markup[["product_id", "product_name", "category", "markup"]]""",
    hint="Use `.assign()` followed by `.nlargest()`.",
)

both_md(
    r"""
## 3. Cleaning: missing values, strings, types, dates, duplicates

Cleaning is not a ritual. For each column, decide:

1. What does missing mean?
2. What dtype matches the domain?
3. What values are valid?
4. What constitutes a duplicate at this table's grain?
"""
)

exercise(
    "Normalize text",
    "On a copy of `customers`, strip and lowercase `email`. Create `full_name` from first and last names with one space between them.",
    """# YOUR CODE HERE\ncustomers_clean = customers.copy()\ncustomers_clean[\"email\"] = ...\ncustomers_clean[\"full_name\"] = ...\ncustomers_clean[[\"email\", \"full_name\"]].head()""",
    """customers_clean = customers.copy()\ncustomers_clean["email"] = customers_clean["email"].str.strip().str.lower()\ncustomers_clean["full_name"] = customers_clean["first_name"].str.cat(customers_clean["last_name"], sep=" ")\ncustomers_clean[["email", "full_name"]].head()""",
    hint="Chain `.str.strip().str.lower()`; use `.str.cat(..., sep=' ')`.",
)

exercise(
    "Handle missing values intentionally",
    "On `customers_clean`, fill missing city with `Unknown`. Fill missing birth year with the median birth year *within each segment*, then convert it to nullable integer `Int64`.",
    """# YOUR CODE HERE\ncustomers_clean[\"city\"] = ...\nsegment_median = ...\ncustomers_clean[\"birth_year\"] = ...\naudit(customers_clean.loc[:, [\"city\", \"birth_year\"]])""",
    """customers_clean["city"] = customers_clean["city"].fillna("Unknown")\nsegment_median = customers_clean.groupby("segment", observed=True)["birth_year"].transform("median")\ncustomers_clean["birth_year"] = customers_clean["birth_year"].fillna(segment_median).round().astype("Int64")\naudit(customers_clean.loc[:, ["city", "birth_year"]])""",
    hint="`groupby(...)[col].transform('median')` returns a Series aligned to the original rows.",
)

exercise(
    "Parse messy values safely",
    "Create the messy Series below. Convert it to numbers so invalid values become missing, then compute its mean.",
    """messy_amounts = pd.Series([\"10.5\", \" 20 \", \"N/A\", \"bad\", None, \"40\"])\n# YOUR CODE HERE\nclean_amounts = ...\nclean_mean = ...\nclean_amounts, clean_mean""",
    """messy_amounts = pd.Series(["10.5", " 20 ", "N/A", "bad", None, "40"])\nclean_amounts = pd.to_numeric(messy_amounts, errors="coerce")\nclean_mean = clean_amounts.mean()\nclean_amounts, clean_mean""",
    hint="Use `pd.to_numeric(..., errors='coerce')`.",
    expected="Mean ≈ 23.50",
)

exercise(
    "Engineer date features",
    "From `orders.order_ts`, create `order_date`, monthly period `order_month`, weekday name, and `is_weekend` on a copy named `orders_enriched`.",
    """# YOUR CODE HERE\norders_enriched = orders.copy()\norders_enriched = orders_enriched.assign(\n    order_date=...,\n    order_month=...,\n    weekday=...,\n    is_weekend=...,\n)\norders_enriched.head()""",
    """orders_enriched = orders.copy()\norders_enriched = orders_enriched.assign(\n    order_date=orders_enriched["order_ts"].dt.normalize(),\n    order_month=orders_enriched["order_ts"].dt.to_period("M"),\n    weekday=orders_enriched["order_ts"].dt.day_name(),\n    is_weekend=orders_enriched["order_ts"].dt.dayofweek.ge(5),\n)\norders_enriched.head()""",
    hint="Use the `.dt` accessor: `normalize`, `to_period`, `day_name`, `dayofweek`.",
)

exercise(
    "Find duplicates at the correct grain",
    "Verify that `item_id` is unique. Then count how many order-product combinations repeat. Explain why repeated `order_id` values alone are expected in `order_items`.",
    """# YOUR CODE HERE\nitem_id_duplicates = ...\norder_product_duplicates = ...\nitem_id_duplicates, order_product_duplicates""",
    """item_id_duplicates = order_items.duplicated("item_id").sum()\norder_product_duplicates = order_items.duplicated(["order_id", "product_id"]).sum()\nitem_id_duplicates, order_product_duplicates\n# Repeated order_id is expected because one order contains multiple line items.""",
    hint="Use `.duplicated(subset).sum()` with the candidate key(s).",
)

exercise(
    "Categoricals and memory",
    "Compare the deep memory use of `orders.channel` before and after conversion to category. Store the two byte counts in `channel_memory`.",
    """# YOUR CODE HERE\nchannel_memory = {\n    \"object_bytes\": ...,\n    \"category_bytes\": ...,\n}\nchannel_memory""",
    """channel_memory = {\n    "object_bytes": orders["channel"].memory_usage(deep=True),\n    "category_bytes": orders["channel"].astype("category").memory_usage(deep=True),\n}\nchannel_memory""",
    hint="Use `.memory_usage(deep=True)`.",
)

both_md(
    r"""
## 4. Feature engineering and method chaining

Prefer vectorized expressions and readable chains. Use `.assign()` for new columns, `.pipe()` for named transformations, and break long chains across lines. `apply(axis=1)` is often a slow Python loop in disguise.
"""
)

exercise(
    "Build the analysis-ready line-item table",
    "Merge `order_items` with selected order and product columns. Keep only completed orders. Add `gross_sales`, `discount_amount`, `net_sales`, `cost`, and `profit`. Validate each merge as many-to-one.",
    """# YOUR CODE HERE\nline_items = (\n    order_items\n    # merge orders\n    # merge products\n    # filter completed\n    # add financial columns\n)\nline_items.head()""",
    """line_items = (\n    order_items\n    .merge(\n        orders[["order_id", "customer_id", "order_ts", "channel", "status"]],\n        on="order_id", how="left", validate="many_to_one"\n    )\n    .merge(\n        products[["product_id", "product_name", "category", "unit_cost"]],\n        on="product_id", how="left", validate="many_to_one"\n    )\n    .loc[lambda d: d["status"].eq("completed")]\n    .assign(\n        gross_sales=lambda d: d["quantity"] * d["unit_price"],\n        discount_amount=lambda d: d["quantity"] * d["unit_price"] * d["discount_pct"],\n        net_sales=lambda d: d["gross_sales"] - d["discount_amount"],\n        cost=lambda d: d["quantity"] * d["unit_cost"],\n        profit=lambda d: d["net_sales"] - d["cost"],\n    )\n)\nline_items.head()""",
    hint="Merge order_items→orders and result→products with `validate='many_to_one'`. In `.assign`, later lambdas can use columns created earlier.",
)

exercise(
    "Replace row-wise apply with vectorization",
    "Create `discount_band` on a copy of `line_items`: `none` for 0, `light` through 5%, `medium` through 10%, and `heavy` above 10%. Use `pd.cut`, not `apply(axis=1)`.",
    """items_banded = line_items.copy()\n# YOUR CODE HERE\nitems_banded[\"discount_band\"] = ...\nitems_banded[\"discount_band\"].value_counts(sort=False)""",
    """items_banded = line_items.copy()\nitems_banded["discount_band"] = pd.cut(\n    items_banded["discount_pct"],\n    bins=[-0.001, 0, 0.05, 0.10, 1],\n    labels=["none", "light", "medium", "heavy"],\n)\nitems_banded["discount_band"].value_counts(sort=False)""",
    hint="Provide ordered bin edges and four labels to `pd.cut`.",
)

exercise(
    "Reusable pipe function",
    "Write `add_margin_metrics(df)` that returns the input with `margin_pct = profit / net_sales` and `is_loss = profit < 0`. Apply it to `line_items` using `.pipe()`.",
    """def add_margin_metrics(df):\n    # YOUR CODE HERE\n    ...\n\nitems_with_margin = ...\nitems_with_margin[[\"net_sales\", \"profit\", \"margin_pct\", \"is_loss\"]].head()""",
    """def add_margin_metrics(df):\n    return df.assign(\n        margin_pct=lambda d: d["profit"].div(d["net_sales"]).where(d["net_sales"].ne(0)),\n        is_loss=lambda d: d["profit"].lt(0),\n    )\n\nitems_with_margin = line_items.pipe(add_margin_metrics)\nitems_with_margin[["net_sales", "profit", "margin_pct", "is_loss"]].head()""",
    hint="Return `df.assign(...)`, then call `line_items.pipe(your_function)`.",
)

both_md(
    r"""
## 5. GroupBy: split, apply, combine

Before grouping, say the output grain aloud: “one row per ___.”

- `.agg()` reduces groups
- `.transform()` returns one value per original row
- `.filter()` keeps or removes whole groups
- named aggregation gives clean output column names
"""
)

exercise(
    "Named aggregation",
    "Create one row per category with total revenue, total profit, units sold, order count, and average discount. Sort by revenue descending.",
    """# YOUR CODE HERE\ncategory_kpis = ...\ncategory_kpis""",
    """category_kpis = (\n    line_items.groupby("category", observed=True)\n    .agg(\n        revenue=("net_sales", "sum"),\n        profit=("profit", "sum"),\n        units=("quantity", "sum"),\n        orders=("order_id", "nunique"),\n        avg_discount=("discount_pct", "mean"),\n    )\n    .sort_values("revenue", ascending=False)\n    .reset_index()\n)\ncategory_kpis""",
    hint="Use `.groupby(...).agg(new_name=(source_column, 'function'))`.",
)

exercise(
    "Transform: share within group",
    "For every line item, calculate its share of the order's net sales. Confirm that shares sum to 1 per order (within floating-point tolerance).",
    """items_share = line_items.copy()\n# YOUR CODE HERE\nitems_share[\"order_revenue\"] = ...\nitems_share[\"share_of_order\"] = ...\nshare_check = ...\nshare_check.head()""",
    """items_share = line_items.copy()\nitems_share["order_revenue"] = items_share.groupby("order_id")["net_sales"].transform("sum")\nitems_share["share_of_order"] = items_share["net_sales"] / items_share["order_revenue"]\nshare_check = items_share.groupby("order_id")["share_of_order"].sum()\nshare_check.head()\n# Optional: np.allclose(share_check, 1.0)""",
    hint="`groupby('order_id')['net_sales'].transform('sum')` aligns totals back to rows.",
)

exercise(
    "Multi-key grouping",
    "Create monthly revenue by channel. Use a monthly period derived from `order_ts`. Return a tidy DataFrame sorted by month and revenue descending within month.",
    """# YOUR CODE HERE\nmonthly_channel = ...\nmonthly_channel.head(10)""",
    """monthly_channel = (\n    line_items.assign(month=lambda d: d["order_ts"].dt.to_period("M"))\n    .groupby(["month", "channel"], as_index=False)\n    .agg(revenue=("net_sales", "sum"))\n    .sort_values(["month", "revenue"], ascending=[True, False])\n)\nmonthly_channel.head(10)""",
    hint="Create month before grouping, then group on `['month', 'channel']`.",
)

exercise(
    "Filter whole groups",
    "Keep all rows belonging to customers whose completed lifetime revenue exceeds 10,000. How many customers and rows remain?",
    """# YOUR CODE HERE\nhigh_value_rows = ...\nresult = ...\nresult""",
    """high_value_rows = line_items.groupby("customer_id").filter(lambda g: g["net_sales"].sum() > 10_000)\nresult = {\n    "customers": high_value_rows["customer_id"].nunique(),\n    "rows": len(high_value_rows),\n}\nresult""",
    hint="Use `groupby(...).filter(lambda group: condition)`.",
)

exercise(
    "Frequency tables",
    "Build a normalized cross-tab of order `channel` by `status`, where each row sums to 1.",
    """# YOUR CODE HERE\nchannel_status = ...\nchannel_status""",
    """channel_status = pd.crosstab(orders["channel"], orders["status"], normalize="index")\nchannel_status""",
    hint="Use `pd.crosstab(..., normalize='index')`.",
)

both_md(
    r"""
## 6. Combining tables: concat, merge, join

Think in database terms:

- identify the key(s)
- state the relationship: one-to-one, one-to-many, or many-to-many
- choose inner/left/right/outer intentionally
- validate cardinality with `validate=`
- use `indicator=True` to audit matches
- check row counts before and after
"""
)

exercise(
    "Merge and preserve grain",
    "Attach customer segment and city to `orders` with a left merge. Validate many-to-one. Verify that row count and unique order count did not change.",
    """# YOUR CODE HERE\norders_customers = ...\nvalidation = ...\nvalidation""",
    """orders_customers = orders.merge(\n    customers[["customer_id", "segment", "city"]],\n    on="customer_id", how="left", validate="many_to_one"\n)\nvalidation = {\n    "same_rows": len(orders_customers) == len(orders),\n    "same_orders": orders_customers["order_id"].nunique() == orders["order_id"].nunique(),\n}\nvalidation""",
    hint="`orders` has many rows per customer; `customers` has one row per customer.",
)

exercise(
    "Anti-join",
    "Find customers who placed no orders. Use a left merge with `indicator=True`; return their IDs and names.",
    """# YOUR CODE HERE\ncustomer_order_match = ...\nno_orders = ...\nno_orders""",
    """customer_order_match = customers.merge(\n    orders[["customer_id"]].drop_duplicates(),\n    on="customer_id", how="left", indicator=True, validate="one_to_one"\n)\nno_orders = customer_order_match.loc[\n    customer_order_match["_merge"].eq("left_only"),\n    ["customer_id", "first_name", "last_name"]\n]\nno_orders""",
    hint="Merge against distinct ordered customer IDs, then filter `_merge == 'left_only'`.",
)

exercise(
    "Return flag without inflating rows",
    "Add `was_returned` to `order_items` using `item_id`. The result must have exactly the same row count as `order_items`.",
    """# YOUR CODE HERE\nitems_returns = ...\nitems_returns[\"was_returned\"] = ...\nlen(items_returns), items_returns[\"was_returned\"].sum()""",
    """items_returns = order_items.merge(\n    returns[["item_id", "return_reason"]],\n    on="item_id", how="left", validate="one_to_one"\n)\nitems_returns["was_returned"] = items_returns["return_reason"].notna()\nlen(items_returns), items_returns["was_returned"].sum()""",
    hint="Because `item_id` is unique in both tables, validate `one_to_one`.",
)

exercise(
    "Concatenate partitions",
    "Split `orders` into Jan–Mar and Apr–Jun partitions, then concatenate them back with `ignore_index=True`. Verify equality after sorting by `order_id`.",
    """# YOUR CODE HERE\nfirst_half = ...\nsecond_half = ...\nrecombined = ...\nare_equal = ...\nare_equal""",
    """first_half = orders.loc[orders["order_ts"].dt.month.le(3)]\nsecond_half = orders.loc[orders["order_ts"].dt.month.gt(3)]\nrecombined = pd.concat([first_half, second_half], ignore_index=True)\nare_equal = (\n    recombined.sort_values("order_id").reset_index(drop=True)\n    .equals(orders.sort_values("order_id").reset_index(drop=True))\n)\nare_equal""",
    hint="Use `pd.concat([part1, part2], ignore_index=True)` and `.equals()`.",
)

exercise(
    "Nearest-time join with merge_asof",
    "For every purchase event, attach the most recent earlier order by the same customer. Calculate hours since that order. Inputs must be sorted correctly.",
    """# YOUR CODE HERE\npurchases = ...\norder_times = ...\npurchase_context = ...\npurchase_context[\"hours_since_order\"] = ...\npurchase_context.head()""",
    """purchases = (\n    events.loc[events["event_type"].eq("purchase"), ["event_id", "customer_id", "event_ts"]]\n    .sort_values(["event_ts", "customer_id"])\n)\norder_times = orders[["order_id", "customer_id", "order_ts"]].sort_values(["order_ts", "customer_id"])\npurchase_context = pd.merge_asof(\n    purchases, order_times,\n    left_on="event_ts", right_on="order_ts", by="customer_id",\n    direction="backward"\n)\npurchase_context["hours_since_order"] = (\n    purchase_context["event_ts"] - purchase_context["order_ts"]\n).dt.total_seconds().div(3600)\npurchase_context.head()""",
    hint="Sort primarily on the time key, then call `pd.merge_asof(..., by='customer_id', direction='backward')`.",
    stretch=True,
)

both_md(
    r"""
## 7. Reshaping: wide, long, pivot, melt, explode

Tidy data usually means: one variable per column, one observation per row, one type of observational unit per table. Reshaping changes representation, not underlying facts.
"""
)

exercise(
    "Pivot with aggregation",
    "Create a table with month as rows, category as columns, and summed net sales as values. Fill missing cells with zero and add row/column totals.",
    """# YOUR CODE HERE\nmonth_category_pivot = ...\nmonth_category_pivot""",
    """month_category_pivot = pd.pivot_table(\n    line_items.assign(month=line_items["order_ts"].dt.to_period("M")),\n    index="month", columns="category", values="net_sales",\n    aggfunc="sum", fill_value=0, margins=True, margins_name="Total", observed=True\n)\nmonth_category_pivot""",
    hint="Use `pd.pivot_table` with `aggfunc='sum'`, `fill_value=0`, and `margins=True`.",
)

exercise(
    "Melt wide data to long",
    "Reset the pivot index, remove the Total row, and melt category columns into `category` and `revenue`. Exclude the Total column.",
    """# YOUR CODE HERE\nwide_no_totals = ...\nlong_revenue = ...\nlong_revenue.head()""",
    """wide_no_totals = month_category_pivot.drop(index="Total").drop(columns="Total").reset_index()\nlong_revenue = wide_no_totals.melt(\n    id_vars="month", var_name="category", value_name="revenue"\n)\nlong_revenue.head()""",
    hint="Clean the margins first, then `.melt(id_vars='month', ...)`.",
)

exercise(
    "Unstack a grouped Series",
    "Count orders by status and channel using `groupby().size()`, then unstack channel into columns and fill missing values with zero.",
    """# YOUR CODE HERE\nstatus_channel_counts = ...\nstatus_channel_counts""",
    """status_channel_counts = (\n    orders.groupby(["status", "channel"]).size().unstack("channel", fill_value=0)\n)\nstatus_channel_counts""",
    hint="`groupby([...]).size()` produces a MultiIndex Series; `.unstack()` moves one level to columns.",
)

exercise(
    "Explode list-like values",
    "Create the small `campaigns` DataFrame below, explode `channels` so each channel gets one row, then count campaigns per channel.",
    """campaigns = pd.DataFrame({\n    \"campaign\": [\"Launch\", \"Ramadan\", \"BackToSchool\"],\n    \"channels\": [[\"Web\", \"App\"], [\"Web\", \"Store\", \"App\"], [\"Store\"]],\n})\n# YOUR CODE HERE\ncampaign_channels = ...\nchannel_counts = ...\nchannel_counts""",
    """campaigns = pd.DataFrame({\n    "campaign": ["Launch", "Ramadan", "BackToSchool"],\n    "channels": [["Web", "App"], ["Web", "Store", "App"], ["Store"]],\n})\ncampaign_channels = campaigns.explode("channels", ignore_index=True)\nchannel_counts = campaign_channels["channels"].value_counts()\nchannel_counts""",
    hint="Use `.explode('channels', ignore_index=True)`.",
)

both_md(
    r"""
## 8. Time series, windows, ranking

Window operations preserve row-level context while looking backward or within groups:

- `shift` → previous/next row
- `diff` / `pct_change` → change from prior row
- `rolling` → moving window
- `expanding` → all observations so far
- `rank` → relative position
"""
)

exercise(
    "Daily resampling and rolling average",
    "Create daily completed revenue indexed by date, including zero-revenue dates, then calculate a 7-day rolling mean with at least one observation.",
    """# YOUR CODE HERE\ndaily_revenue = ...\nrolling_7d = ...\npd.DataFrame({\"revenue\": daily_revenue, \"rolling_7d\": rolling_7d}).head(10)""",
    """daily_revenue = (\n    line_items.set_index("order_ts")["net_sales"]\n    .resample("D").sum()\n)\nrolling_7d = daily_revenue.rolling(7, min_periods=1).mean()\npd.DataFrame({"revenue": daily_revenue, "rolling_7d": rolling_7d}).head(10)""",
    hint="Set a DatetimeIndex, `.resample('D').sum()`, then `.rolling(7, min_periods=1).mean()`.",
)

exercise(
    "Previous order with shift",
    "At one row per completed order, calculate each customer's previous order timestamp and days since previous order.",
    """# YOUR CODE HERE\norder_level = ...\norder_level[\"previous_order_ts\"] = ...\norder_level[\"days_since_previous\"] = ...\norder_level.head(10)""",
    """order_level = (\n    line_items.groupby(["order_id", "customer_id", "order_ts"], as_index=False)\n    .agg(order_revenue=("net_sales", "sum"))\n    .sort_values(["customer_id", "order_ts"])\n)\norder_level["previous_order_ts"] = order_level.groupby("customer_id")["order_ts"].shift()\norder_level["days_since_previous"] = (\n    order_level["order_ts"] - order_level["previous_order_ts"]\n).dt.total_seconds().div(86_400)\norder_level.head(10)""",
    hint="Sort within customer before `groupby('customer_id')['order_ts'].shift()`.",
)

exercise(
    "Cumulative customer value",
    "On `order_level`, create each customer's cumulative revenue and sequential order number.",
    """# YOUR CODE HERE\norder_level[\"cumulative_revenue\"] = ...\norder_level[\"order_number\"] = ...\norder_level.head(10)""",
    """order_level["cumulative_revenue"] = order_level.groupby("customer_id")["order_revenue"].cumsum()\norder_level["order_number"] = order_level.groupby("customer_id").cumcount().add(1)\norder_level.head(10)""",
    hint="Use grouped `.cumsum()` and `groupby(...).cumcount() + 1`.",
)

exercise(
    "Rank products within category",
    "Aggregate product revenue, then rank products by revenue within category. Keep the top 3 per category. Break ties with first appearance order.",
    """# YOUR CODE HERE\nproduct_revenue = ...\nproduct_revenue[\"category_rank\"] = ...\ntop3_per_category = ...\ntop3_per_category""",
    """product_revenue = (\n    line_items.groupby(["category", "product_id", "product_name"], observed=True, as_index=False)\n    .agg(revenue=("net_sales", "sum"))\n    .sort_values(["category", "revenue"], ascending=[True, False])\n)\nproduct_revenue["category_rank"] = product_revenue.groupby("category", observed=True)["revenue"].rank(\n    method="first", ascending=False\n)\ntop3_per_category = product_revenue.loc[product_revenue["category_rank"].le(3)]\ntop3_per_category""",
    hint="Aggregate first, then `groupby('category')['revenue'].rank(method='first', ascending=False)`.",
)

exercise(
    "Customer value quartiles",
    "Aggregate completed revenue per customer and assign quartiles Q1–Q4 with `qcut`, where Q4 is highest value.",
    """# YOUR CODE HERE\ncustomer_value = ...\ncustomer_value[\"value_quartile\"] = ...\ncustomer_value.sort_values(\"lifetime_revenue\", ascending=False).head()""",
    """customer_value = (\n    line_items.groupby("customer_id", as_index=False)\n    .agg(lifetime_revenue=("net_sales", "sum"))\n)\ncustomer_value["value_quartile"] = pd.qcut(\n    customer_value["lifetime_revenue"], q=4, labels=["Q1", "Q2", "Q3", "Q4"]\n)\ncustomer_value.sort_values("lifetime_revenue", ascending=False).head()""",
    hint="Use `pd.qcut(series, q=4, labels=[...])`.",
)

both_md(
    r"""
## 9. Debugging and performance habits

The fastest pandas users are not the ones who remember the most methods. They detect wrong grain, silent row multiplication, dtype drift, and accidental mutation early.
"""
)

exercise(
    "Fix chained assignment",
    "The intended result is a copy of orders where high shipping fees are labeled `expensive`. Rewrite the broken pattern safely: `df[df['shipping_fee'] > 15]['shipping_label'] = 'expensive'`.",
    """orders_debug = orders.copy()\n# YOUR FIX HERE\n...\norders_debug[\"shipping_label\"].value_counts(dropna=False)""",
    """orders_debug = orders.copy()\norders_debug["shipping_label"] = "regular"\norders_debug.loc[orders_debug["shipping_fee"].gt(15), "shipping_label"] = "expensive"\norders_debug["shipping_label"].value_counts(dropna=False)""",
    hint="Assign with `.loc[mask, 'shipping_label']` on the original copy.",
)

exercise(
    "Diagnose a many-to-many merge",
    "Create `customer_categories` as distinct customer-category pairs from `line_items`. Predict why merging it to `orders` only on `customer_id` increases rows. Perform the merge and quantify the inflation factor.",
    """customer_categories = ...\n# Predict before running!\norders_expanded = ...\ninflation_factor = ...\ninflation_factor""",
    """customer_categories = line_items[["customer_id", "category"]].drop_duplicates()\norders_expanded = orders.merge(\n    customer_categories, on="customer_id", how="left", validate="many_to_many"\n)\ninflation_factor = len(orders_expanded) / len(orders)\ninflation_factor\n# Each customer can have several categories, so each order repeats once per customer-category.""",
    hint="The right table is not unique on `customer_id`; inspect `.duplicated('customer_id')`.",
)

exercise(
    "Use eval for a vectorized expression",
    "Recreate gross sales and discounted net sales from `order_items` with `.eval()`. Compare the result to an ordinary vectorized expression using `np.allclose`.",
    """evaluated = order_items.copy()\n# YOUR CODE HERE\nevaluated = ...\nsame_result = ...\nsame_result""",
    """evaluated = order_items.copy().eval(\n    "gross = quantity * unit_price\\nnet = gross * (1 - discount_pct)"\n)\nsame_result = np.allclose(\n    evaluated["net"],\n    order_items["quantity"] * order_items["unit_price"] * (1 - order_items["discount_pct"])\n)\nsame_result""",
    hint="Pass a multiline expression string to `.eval()`.",
    stretch=True,
)

exercise(
    "Round-trip a typed file",
    "Write `line_items` to a temporary Parquet file and read it back. Confirm shape and dtypes. This avoids polluting your workspace.",
    """import tempfile\nfrom pathlib import Path\n\n# YOUR CODE HERE\nwith tempfile.TemporaryDirectory() as tmp:\n    path = ...\n    ...\n    restored = ...\n\nroundtrip_ok = ...\nroundtrip_ok""",
    """import tempfile\nfrom pathlib import Path\n\nwith tempfile.TemporaryDirectory() as tmp:\n    path = Path(tmp) / "line_items.parquet"\n    line_items.to_parquet(path, index=False)\n    restored = pd.read_parquet(path)\n\nroundtrip_ok = {\n    "same_shape": restored.shape == line_items.shape,\n    "same_dtypes": restored.dtypes.equals(line_items.dtypes),\n}\nroundtrip_ok""",
    hint="Use `.to_parquet(path, index=False)` and `pd.read_parquet(path)`.",
    stretch=True,
)

both_md(
    r"""
## 10. Capstone: six-month commerce review

You are the analyst for this retailer. Leadership wants a compact performance review using completed orders only. Build each table from the source data, even if a similar intermediate exists above.

**Quality bar**

- every table has a stated grain
- merges use `validate=`
- financial columns are explicit
- missing values are handled intentionally
- tables are sorted for human reading
- your conclusions cite numbers
"""
)

exercise(
    "Capstone A — analysis base",
    "Build `capstone_items`: one row per completed order item, with customer segment/city, product category/cost, and all financial columns. Add a boolean return flag without changing the grain.",
    """# YOUR CODE HERE\ncapstone_items = ...\n# Add assertions that protect the grain.\n...\ncapstone_items.head()""",
    """capstone_items = (\n    order_items\n    .merge(orders, on="order_id", how="left", validate="many_to_one")\n    .merge(products, on="product_id", how="left", validate="many_to_one")\n    .merge(customers[["customer_id", "segment", "city"]], on="customer_id", how="left", validate="many_to_one")\n    .merge(returns[["item_id", "return_reason"]], on="item_id", how="left", validate="one_to_one")\n    .loc[lambda d: d["status"].eq("completed")]\n    .assign(\n        gross_sales=lambda d: d["quantity"] * d["unit_price"],\n        discount_amount=lambda d: d["gross_sales"] * d["discount_pct"],\n        net_sales=lambda d: d["gross_sales"] - d["discount_amount"],\n        cost=lambda d: d["quantity"] * d["unit_cost"],\n        profit=lambda d: d["net_sales"] - d["cost"],\n        was_returned=lambda d: d["return_reason"].notna(),\n    )\n)\nassert capstone_items["item_id"].is_unique\nassert capstone_items["order_id"].nunique() == orders.loc[orders["status"].eq("completed"), "order_id"].nunique()\ncapstone_items.head()""",
    hint="Merge items→orders→products→customers→returns. The relationship is many-to-one except returns on unique item ID.",
)

exercise(
    "Capstone B — executive KPI card",
    "Produce a one-row DataFrame with revenue, profit, margin %, completed orders, customers, units, average order value, and item return rate.",
    """# YOUR CODE HERE\nexecutive_kpis = ...\nexecutive_kpis""",
    """total_revenue = capstone_items["net_sales"].sum()\ntotal_profit = capstone_items["profit"].sum()\ncompleted_orders = capstone_items["order_id"].nunique()\nexecutive_kpis = pd.DataFrame({\n    "revenue": [total_revenue],\n    "profit": [total_profit],\n    "margin_pct": [total_profit / total_revenue],\n    "completed_orders": [completed_orders],\n    "customers": [capstone_items["customer_id"].nunique()],\n    "units": [capstone_items["quantity"].sum()],\n    "avg_order_value": [total_revenue / completed_orders],\n    "item_return_rate": [capstone_items["was_returned"].mean()],\n})\nexecutive_kpis""",
    hint="AOV uses unique orders, not line-item rows. Return rate here is returned items / items.",
)

exercise(
    "Capstone C — monthly trend",
    "Create monthly revenue, profit, orders, AOV, and month-over-month revenue growth. Plot revenue and profit on one chart.",
    """# YOUR CODE HERE\nmonthly_kpis = ...\n... # calculate AOV and growth\n... # plot\nmonthly_kpis""",
    """monthly_kpis = (\n    capstone_items.assign(month=lambda d: d["order_ts"].dt.to_period("M"))\n    .groupby("month", as_index=False)\n    .agg(\n        revenue=("net_sales", "sum"),\n        profit=("profit", "sum"),\n        orders=("order_id", "nunique"),\n    )\n    .sort_values("month")\n)\nmonthly_kpis["aov"] = monthly_kpis["revenue"] / monthly_kpis["orders"]\nmonthly_kpis["revenue_growth"] = monthly_kpis["revenue"].pct_change()\nmonthly_kpis.assign(month=monthly_kpis["month"].astype(str)).plot(\n    x="month", y=["revenue", "profit"], marker="o", figsize=(9, 4), title="Monthly performance"\n)\nplt.xticks(rotation=45)\nplt.tight_layout()\nplt.show()\nmonthly_kpis""",
    hint="Group by a monthly Period, then calculate AOV and `.pct_change()` after sorting.",
)

exercise(
    "Capstone D — category scorecard",
    "Create one row per category with revenue, profit, margin %, return rate, units, and revenue share. Rank categories by revenue.",
    """# YOUR CODE HERE\ncategory_scorecard = ...\ncategory_scorecard""",
    """category_scorecard = (\n    capstone_items.groupby("category", observed=True, as_index=False)\n    .agg(\n        revenue=("net_sales", "sum"),\n        profit=("profit", "sum"),\n        return_rate=("was_returned", "mean"),\n        units=("quantity", "sum"),\n    )\n    .assign(\n        margin_pct=lambda d: d["profit"] / d["revenue"],\n        revenue_share=lambda d: d["revenue"] / d["revenue"].sum(),\n    )\n    .sort_values("revenue", ascending=False)\n)\ncategory_scorecard["revenue_rank"] = category_scorecard["revenue"].rank(method="dense", ascending=False).astype(int)\ncategory_scorecard""",
    hint="Aggregate first; derive ratios afterward to avoid averaging row-level ratios.",
)

exercise(
    "Capstone E — RFM segmentation",
    "At customer grain, compute recency in days from 2025-07-01, completed-order frequency, and monetary value. Score each metric 1–4 with quantiles and create an RFM string such as `444`.",
    """snapshot_date = pd.Timestamp(\"2025-07-01\")\n# YOUR CODE HERE\nrfm = ...\n# score recency (lower is better), frequency and monetary (higher are better)\n...\nrfm.sort_values([\"r_score\", \"f_score\", \"m_score\"], ascending=False).head(10)""",
    """snapshot_date = pd.Timestamp("2025-07-01")\nrfm = (\n    capstone_items.groupby("customer_id", as_index=False)\n    .agg(\n        last_order=("order_ts", "max"),\n        frequency=("order_id", "nunique"),\n        monetary=("net_sales", "sum"),\n    )\n)\nrfm["recency"] = (snapshot_date - rfm["last_order"].dt.normalize()).dt.days\nrfm["r_score"] = pd.qcut(rfm["recency"].rank(method="first"), 4, labels=[4, 3, 2, 1]).astype(int)\nrfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)\nrfm["m_score"] = pd.qcut(rfm["monetary"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)\nrfm["rfm_segment"] = rfm[["r_score", "f_score", "m_score"]].astype(str).agg("".join, axis=1)\nrfm.sort_values(["r_score", "f_score", "m_score"], ascending=False).head(10)""",
    hint="Aggregate last order, unique order count, and revenue. Rank before `qcut` to avoid duplicate bin-edge problems.",
    stretch=True,
)

exercise(
    "Capstone F — cohort retention",
    "Define each customer's acquisition month as their first completed order month. Build a cohort table of unique active customers by months since acquisition, then divide by cohort size to get retention.",
    """# YOUR CODE HERE\ncustomer_months = ...\n# acquisition_month, activity_month, period_number\ncohort_counts = ...\nretention = ...\nretention""",
    """customer_months = (\n    capstone_items.assign(activity_month=lambda d: d["order_ts"].dt.to_period("M"))\n    [["customer_id", "activity_month"]]\n    .drop_duplicates()\n)\ncustomer_months["acquisition_month"] = customer_months.groupby("customer_id")["activity_month"].transform("min")\ncustomer_months["period_number"] = (\n    customer_months["activity_month"].astype("int64")\n    - customer_months["acquisition_month"].astype("int64")\n)\ncohort_counts = customer_months.pivot_table(\n    index="acquisition_month", columns="period_number", values="customer_id", aggfunc="nunique", fill_value=0\n)\nretention = cohort_counts.div(cohort_counts[0], axis=0)\nretention""",
    hint="Reduce to unique customer-months. Convert monthly Periods to integer ordinals to subtract them.",
    stretch=True,
)

exercise(
    "Capstone G — investigate return risk",
    "Compare return rates by category and discount band. Require at least 20 items per cell, and sort highest risk first.",
    """# YOUR CODE HERE\nreturn_risk = ...\nreturn_risk""",
    """return_risk = (\n    capstone_items.assign(\n        discount_band=pd.cut(\n            capstone_items["discount_pct"],\n            bins=[-0.001, 0, 0.05, 0.10, 1],\n            labels=["none", "light", "medium", "heavy"],\n        )\n    )\n    .groupby(["category", "discount_band"], observed=True, as_index=False)\n    .agg(items=("item_id", "size"), return_rate=("was_returned", "mean"))\n    .loc[lambda d: d["items"].ge(20)]\n    .sort_values("return_rate", ascending=False)\n)\nreturn_risk""",
    hint="Bin discounts, aggregate item count and mean of the boolean return flag, then filter reliable cells.",
)

exercise(
    "Capstone H — write the decision memo",
    "Write 4–6 bullets in a Markdown cell: overall performance, strongest/weakest category, trend, customer insight, return-risk insight, and one action. Every claim must cite a number from your tables.",
    """# Optional helper: format the tables you need, then write your memo in a Markdown cell.\ndisplay(executive_kpis)\ndisplay(monthly_kpis)\ndisplay(category_scorecard)\ndisplay(return_risk.head())""",
    """display(executive_kpis)\ndisplay(monthly_kpis)\ndisplay(category_scorecard)\ndisplay(return_risk.head())\n# The memo is intentionally open-ended. A strong answer cites calculated values,\n# distinguishes facts from interpretations, and recommends one measurable action.""",
)

both_md(
    r"""
## 11. Final recall drill — do this without running code

Write the pandas expression you would reach for first:

1. Select rows by condition and two named columns.
2. Add a group total back to every original row.
3. Attach one customer record to many orders safely.
4. Find left-table records with no right-table match.
5. Convert a wide table to tidy long form.
6. Get the prior timestamp within each customer.
7. Calculate daily totals from timestamps.
8. Rank products within category.
9. Detect whether a supposed key is duplicated.
10. Conditionally update a column without chained assignment.

<details><summary>Compact recall key</summary>

1. `.loc[mask, [col1, col2]]`
2. `.groupby(key)[value].transform('sum')`
3. `.merge(..., validate='many_to_one')`
4. `.merge(..., indicator=True)` then filter `left_only`
5. `.melt(...)`
6. sort, then `.groupby(key)[time].shift()`
7. set a DatetimeIndex, then `.resample('D').sum()`
8. `.groupby(category)[value].rank(...)`
9. `.duplicated(key).any()`
10. `.loc[mask, col] = value`

</details>
"""
)

both_md(
    r"""
## Pandas decision cheat sheet

| Need | Reach for |
|---|---|
| Inspect shape/types/missingness | `.shape`, `.info()`, `.isna().sum()`, custom audit |
| Pick rows/columns | `.loc`, `.iloc`, `query` |
| Create columns | `.assign`, vectorized operators, `np.select`, `cut` |
| Clean text | `.str` methods |
| Work with dates | `pd.to_datetime`, `.dt`, `resample` |
| Reduce each group | `groupby().agg()` |
| Add group facts to rows | `groupby().transform()` |
| Keep/remove whole groups | `groupby().filter()` |
| Combine columns from tables | `merge(..., validate=...)` |
| Stack row partitions | `concat` |
| Wide ↔ long | `pivot_table`, `melt`, `stack`, `unstack` |
| Previous/moving/cumulative | `shift`, `rolling`, `expanding`, `cumsum` |
| Top-N | `nlargest`, `nsmallest`, grouped `rank` |
| Validate assumptions | `assert`, `.duplicated`, row counts, merge indicator |

### The five questions to ask before every analysis

1. What does one row represent right now?
2. What should one row represent after this operation?
3. Which columns form the key?
4. Can this operation duplicate or drop rows?
5. Are my dtypes and missing values compatible with the question?

### Tomorrow's 25-minute retention loop

- Rebuild `line_items` from memory.
- Re-solve one GroupBy, one merge, one reshape, and one window exercise.
- Explain `agg` versus `transform` aloud.
- Recreate the executive KPI card without looking.

If you can complete the capstone cleanly and explain your grain at each step, you have closed the practical pandas gap.
"""
)


def notebook(cells: list[dict], title: str) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
            "title": title,
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


LEARNER.write_text(json.dumps(notebook(learner_cells, "Pandas Learning by Doing"), indent=1), encoding="utf-8")
SOLUTIONS.write_text(json.dumps(notebook(solution_cells, "Pandas Learning by Doing — Solutions"), indent=1), encoding="utf-8")
print(f"Wrote {LEARNER.name}: {len(learner_cells)} cells")
print(f"Wrote {SOLUTIONS.name}: {len(solution_cells)} cells")
print(f"Exercises: {exercise_counter}")
