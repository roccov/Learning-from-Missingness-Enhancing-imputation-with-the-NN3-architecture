import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from dateutil.relativedelta import relativedelta


def sliding_window_lasso_single_alpha(
    df: pd.DataFrame,
    date_col: str,
    features: list,
    target: str,
    alpha: float,
    train_years: int = 25,
    test_years: int = 1,
    winsor_limits: tuple = (1, 99),
    date_sorted: bool = True
) -> (pd.DataFrame, pd.DataFrame):
    """
    Apply Lasso with a fixed alpha on sliding windows (all windows).

    Returns:
      - metrics_df: one row per window with alpha, r2_train, r2_test, n_nonzero, selected_features
      - preds_df: test-set predictions with date, permno, y_true, y_pred, window_start
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    if not date_sorted:
        df = df.sort_values(date_col)

    start_date = df[date_col].min()
    end_date = df[date_col].max()

    metrics = []
    preds = []
    current_start = start_date

    while True:
        train_start = current_start
        train_end = train_start + relativedelta(years=train_years)
        test_end = train_end + relativedelta(years=test_years)
        if test_end > end_date:
            break

        # Slice into train/test
        train_df = df[(df[date_col] >= train_start) & (df[date_col] < train_end)].copy()
        test_df = df[(df[date_col] >= train_end) & (df[date_col] < test_end)].copy()

        # Winsorize features
        low, high = winsor_limits
        for col in features:
            bounds = np.percentile(train_df[col].dropna(), [low, high])
            train_df[col] = train_df[col].clip(*bounds)
            test_df[col] = test_df[col].clip(*bounds)

        # Standardize
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_df[features])
        X_test = scaler.transform(test_df[features])
        y_train = train_df[target].values
        y_test = test_df[target].values

        # Fit Lasso with fixed alpha
        model = Lasso(alpha=alpha, max_iter=20000, tol=1e-4)
        model.fit(X_train, y_train)

        # Metrics
        r2_train = model.score(X_train, y_train)
        r2_test = model.score(X_test, y_test)
        coef = model.coef_
        n_nonzero = np.sum(coef != 0)
        selected = [features[i] for i, c in enumerate(coef) if c != 0]
        metrics.append({
            'train_start': train_start,
            'train_end': train_end,
            'test_end': test_end,
            'alpha': alpha,
            'r2_train': r2_train,
            'r2_test': r2_test,
            'n_nonzero': n_nonzero,
            'selected_features': selected
        })

        # Test predictions
        y_pred = model.predict(X_test)
        preds.append(pd.DataFrame({
            date_col: test_df[date_col].values,
            'permno': test_df['permno'].values,
            'y_true': y_test,
            'y_pred': y_pred,
            'window_start': train_start
        }))

        # Advance window
        current_start += relativedelta(years=1)

    metrics_df = pd.DataFrame(metrics)
    preds_df = pd.concat(preds, ignore_index=True)
    return metrics_df, preds_df


def main():
    # Load data
    df = pd.read_csv('crsp_compustat_jkp_ExRet_withFlag_noWins_noStd.csv')

    # Common parameters
    date_col = 'date'
    target_col = 'Excess_ret_target'
    exclude = {date_col, 'permno', 'Unnamed: 0', 'Sp_ret', target_col, 'ret'}
    features = [c for c in df.columns if c not in exclude and not c.endswith('_flag')]

    # List of alphas to test
    alphas = [0.0005,0.001,0.0015,0.002,0.0025,0.003,0.0035,0.004,0.0045,0.0055]

    for alpha in alphas:
        # Execution on all windows
        metrics_df, preds_df = sliding_window_lasso_single_alpha(
            df,
            date_col=date_col,
            features=features,
            target=target_col,
            alpha=alpha,
            train_years=25,
            test_years=1,
            winsor_limits=(1, 99),
            date_sorted=True
        )

        # R^2 test median calculation
        median_r2_test = metrics_df['r2_test'].median()
        print(f"Median R² test for alpha={alpha}: {median_r2_test:.4f}")

        # Export results and predictions
        metrics_filename = f'results_25_alpha_{alpha}.csv'
        preds_filename = f'predictions_25_alpha_{alpha}.csv'
        metrics_df.to_csv(metrics_filename, index=False)
        preds_df.to_csv(preds_filename, index=False)
        print(f'Exports done : {metrics_filename} et {preds_filename}')

if __name__ == '__main__':
    main()
