"""
데이터 수집 모듈
"""

import os
import ccxt
import pandas as pd
from datetime import datetime
from alpha_vantage.foreignexchange import ForeignExchange
from alpha_vantage.timeseries import TimeSeries
from .utils import get_data_paths

class DataCollector:
    """데이터 수집 클래스"""
    
    def __init__(self, output_dir=None):
        self.paths = get_data_paths()
        self.output_dir = output_dir or self.paths['data_dir']
    
    def get_ccxt_data(self, exchange, base, quote, start_date, end_date):
        """
        CCXT를 사용하여 암호화폐 또는 Forex 데이터를 수집하는 함수

        Parameters:
        - exchange: 거래소 이름 (예: 'kraken', 'oanda')
        - base: 기준 통화 (예: 'BTC', 'USD')
        - quote: 상대 통화 (예: 'USD', 'EUR')
        - start_date: 데이터 시작 날짜 (예: '2023-01-01')
        - end_date: 데이터 종료 날짜 (예: '2025-07-31')

        Returns:
        - pandas DataFrame with 'timestamp', 'close', 'volume' columns
        """ 
        try:
            exchange_class = getattr(ccxt, exchange.lower())
            exchange = exchange_class()
            exchange.load_markets()

            symbol = f"{base}/{quote}"
            if symbol not in exchange.symbols:
                raise ValueError(f"{symbol}은(는) {exchange}에서 지원되지 않습니다.")
            
            timeframe = '1d'
            since = int(pd.to_datetime(start_date).timestamp() * 1000)

            all_data = []
            while True:
                candles = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=215)
                if not candles:
                    break
                df = pd.DataFrame(
                    candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df[['timestamp', 'close', 'volume']]
                all_data.append(df)

                last_timestamp = candles[-1][0]
                since = last_timestamp + 24 * 60 * 60 * 1000

                if pd.to_datetime(last_timestamp, unit='ms') > pd.to_datetime(end_date):
                    break
            
            df = pd.concat(all_data)
            df.set_index('timestamp', inplace=True)
            df = df.rename(columns={
                'close': f'{base}_{quote}',
                'volume': f'{base}_Volume'
            })
            df = df.loc[start_date:end_date]

            csv_file = os.path.join(self.output_dir, f"{exchange}_{base}_{quote}.csv")
            df.to_csv(csv_file, encoding='utf-8')
            print(f"{exchange} {base}/{quote} 데이터 크기: {df.shape}, 저장: {csv_file}")
            
            return df

        except Exception as e:
            print(f"get_ccxt_data({exchange}, {base}/{quote}) 오류: {e}")
            return None

    def get_alpha_vantage_forex_data(self, base, quote, start_date, end_date, api_key):
        """
        Alpha Vantage API를 사용하여 Forex 데이터를 수집하는 함수

        Parameters:
        - base: 기준 통화 (예: 'USD')
        - quote: 상대 통화 (예: 'EUR')
        - start_date: 데이터 시작 날짜 (예: '2023-01-01')
        - end_date: 데이터 종료 날짜 (예: '2025-07-31')
        - api_key: Alpha Vantage API 키

        Returns:
        - pandas DataFrame
        """
        try:
            fx = ForeignExchange(key=api_key)
            data, meta_data = fx.get_currency_exchange_daily(
                from_symbol=base, 
                to_symbol=quote, 
                outputsize='full'
            )
            
            df = pd.DataFrame(data).T
            df.index = pd.to_datetime(df.index)
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df = df.astype(float)
            
            # 날짜 범위 필터링
            df = df.loc[start_date:end_date]
            
            csv_file = os.path.join(self.output_dir, f"alpha_vantage_{base}_{quote}.csv")
            df.to_csv(csv_file, encoding='utf-8')
            print(f"Alpha Vantage {base}/{quote} 데이터 크기: {df.shape}, 저장: {csv_file}")
            
            return df
            
        except Exception as e:
            print(f"get_alpha_vantage_forex_data({base}/{quote}) 오류: {e}")
            return None

    def get_alpha_vantage_gold_data(self, start_date, end_date, api_key):
        """
        Alpha Vantage API를 사용하여 금 데이터를 수집하는 함수

        Parameters:
        - start_date: 데이터 시작 날짜 (예: '2023-01-01')
        - end_date: 데이터 종료 날짜 (예: '2025-07-31')
        - api_key: Alpha Vantage API 키

        Returns:
        - pandas DataFrame
        """
        try:
            ts = TimeSeries(key=api_key)
            data, meta_data = ts.get_daily('XAU/USD', outputsize='full')
            
            df = pd.DataFrame(data).T
            df.index = pd.to_datetime(df.index)
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df = df.astype(float)
            
            # 날짜 범위 필터링
            df = df.loc[start_date:end_date]
            
            csv_file = os.path.join(self.output_dir, "alpha_vantage_XAU_USD.csv")
            df.to_csv(csv_file, encoding='utf-8')
            print(f"Alpha Vantage XAU/USD 데이터 크기: {df.shape}, 저장: {csv_file}")
            
            return df
            
        except Exception as e:
            print(f"get_alpha_vantage_gold_data() 오류: {e}")
            return None

    def collect_crypto_forex_data(self, start_date, end_date, alpha_vantage_api_key, output_file='crypto_forex_data.csv'):
        """
        암호화폐와 Forex 데이터를 수집하고 병합하는 함수

        Parameters:
        - start_date: 데이터 시작 날짜
        - end_date: 데이터 종료 날짜
        - alpha_vantage_api_key: Alpha Vantage API 키
        - output_file: 출력 파일명

        Returns:
        - pandas DataFrame
        """
        try:
            # CCXT 데이터 수집
            btc_eur_data = self.get_ccxt_data('kraken', 'BTC', 'EUR', start_date, end_date)
            btc_jpy_data = self.get_ccxt_data('kraken', 'BTC', 'JPY', start_date, end_date)
            
            # Alpha Vantage 데이터 수집
            usd_eur_data = self.get_alpha_vantage_forex_data('USD', 'EUR', start_date, end_date, alpha_vantage_api_key)
            usd_jpy_data = self.get_alpha_vantage_forex_data('USD', 'JPY', start_date, end_date, alpha_vantage_api_key)
            usd_krw_data = self.get_alpha_vantage_forex_data('USD', 'KRW', start_date, end_date, alpha_vantage_api_key)
            usd_cny_data = self.get_alpha_vantage_forex_data('USD', 'CNY', start_date, end_date, alpha_vantage_api_key)
            xau_usd_data = self.get_alpha_vantage_gold_data(start_date, end_date, alpha_vantage_api_key)
            
            # 데이터 병합
            all_data = []
            
            if btc_eur_data is not None:
                all_data.append(btc_eur_data)
            if btc_jpy_data is not None:
                all_data.append(btc_jpy_data)
            if usd_eur_data is not None:
                all_data.append(usd_eur_data)
            if usd_jpy_data is not None:
                all_data.append(usd_jpy_data)
            if usd_krw_data is not None:
                all_data.append(usd_krw_data)
            if usd_cny_data is not None:
                all_data.append(usd_cny_data)
            if xau_usd_data is not None:
                all_data.append(xau_usd_data)
            
            if all_data:
                merged_df = pd.concat(all_data, axis=1)
                merged_df = merged_df.fillna(method='ffill')
                
                output_path = os.path.join(self.output_dir, output_file)
                merged_df.to_csv(output_path, encoding='utf-8')
                print(f"병합된 데이터 크기: {merged_df.shape}, 저장: {output_path}")
                
                return merged_df
            else:
                print("수집된 데이터가 없습니다.")
                return None
                
        except Exception as e:
            print(f"collect_crypto_forex_data() 오류: {e}")
            return None 