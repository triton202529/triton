#!/usr/bin/env python3
"""
Execution Intelligence Engine for Triton

Implements sophisticated order execution algorithms:
- Market impact modeling (temporary & permanent)
- VWAP/TWAP execution strategies
- Smart order routing
- Slippage prediction and minimization
- Optimal trade scheduling
- Transaction cost analysis (TCA)

Used by institutional traders to minimize execution costs.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class ExecutionIntelligenceEngine:
    """
    Execution Intelligence Engine for optimal trade execution.
    
    Features:
    - Market impact models
    - VWAP/TWAP strategies
    - Smart order routing
    - Slippage prediction
    - Transaction cost analysis
    """
    
    def __init__(self,
                 impact_model: str = 'almgren_chriss',
                 participation_rate: float = 0.10,
                 risk_aversion: float = 1e-6,
                 verbose: bool = False):
        """
        Initialize execution engine.
        
        Args:
            impact_model: 'almgren_chriss', 'linear', or 'square_root'
            participation_rate: Target volume participation rate (0-1)
            risk_aversion: Risk aversion parameter for optimization
            verbose: Enable verbose logging
        """
        self.impact_model = impact_model
        self.participation_rate = participation_rate
        self.risk_aversion = risk_aversion
        self.verbose = verbose
        
        # Execution history
        self.execution_history = []
        
    def _log(self, *args, **kwargs):
        """Logging helper."""
        if self.verbose:
            print(*args, **kwargs)
    
    def estimate_market_impact(self,
                               order_size: float,
                               daily_volume: float,
                               volatility: float,
                               price: float) -> Dict[str, float]:
        """
        Estimate market impact of an order.
        
        Uses industry-standard models to estimate price impact.
        
        Args:
            order_size: Number of shares to trade
            daily_volume: Average daily volume
            volatility: Daily volatility (std of returns)
            price: Current price
        
        Returns:
            Impact estimates (temporary, permanent, total)
        """
        # Volume participation
        participation = order_size / daily_volume if daily_volume > 0 else 1
        
        if self.impact_model == 'almgren_chriss':
            # Almgren-Chriss model
            # Temporary impact: η * (v/V)
            # Permanent impact: γ * (v/V)
            eta = 0.1 * volatility * price  # Temporary impact coefficient
            gamma = 0.05 * volatility * price  # Permanent impact coefficient
            
            temporary_impact = eta * participation
            permanent_impact = gamma * participation
            
        elif self.impact_model == 'square_root':
            # Square root model (common in practice)
            # Impact ∝ sqrt(v/V)
            impact_coef = 0.1 * volatility * price
            sqrt_participation = np.sqrt(participation)
            
            temporary_impact = impact_coef * sqrt_participation * 0.7
            permanent_impact = impact_coef * sqrt_participation * 0.3
            
        else:  # linear
            # Simple linear model
            impact_coef = 0.05 * volatility * price
            temporary_impact = impact_coef * participation * 0.6
            permanent_impact = impact_coef * participation * 0.4
        
        total_impact = temporary_impact + permanent_impact
        
        return {
            'temporary_impact': temporary_impact,
            'permanent_impact': permanent_impact,
            'total_impact': total_impact,
            'total_impact_bps': (total_impact / price) * 10000,  # Basis points
            'participation': participation
        }
    
    def calculate_vwap_schedule(self,
                                total_shares: int,
                                time_horizon: int,
                                volume_profile: Optional[List[float]] = None) -> List[Tuple[int, int]]:
        """
        Calculate VWAP (Volume-Weighted Average Price) execution schedule.
        
        Args:
            total_shares: Total shares to execute
            time_horizon: Number of periods (e.g., minutes or hours)
            volume_profile: Expected volume distribution (optional)
        
        Returns:
            List of (period, shares) tuples
        """
        if volume_profile is None:
            # U-shaped intraday volume pattern (typical)
            volume_profile = self._generate_default_volume_profile(time_horizon)
        
        # Normalize volume profile
        volume_profile = np.array(volume_profile)
        volume_profile = volume_profile / volume_profile.sum()
        
        # Allocate shares proportional to volume
        shares_schedule = (total_shares * volume_profile).astype(int)
        
        # Adjust for rounding
        shortfall = total_shares - shares_schedule.sum()
        if shortfall > 0:
            shares_schedule[-1] += shortfall
        elif shortfall < 0:
            shares_schedule[-1] += shortfall
        
        schedule = [(i, int(shares)) for i, shares in enumerate(shares_schedule) if shares > 0]
        
        return schedule
    
    def calculate_twap_schedule(self,
                                total_shares: int,
                                time_horizon: int) -> List[Tuple[int, int]]:
        """
        Calculate TWAP (Time-Weighted Average Price) execution schedule.
        
        Args:
            total_shares: Total shares to execute
            time_horizon: Number of periods
        
        Returns:
            List of (period, shares) tuples
        """
        shares_per_period = total_shares // time_horizon
        remainder = total_shares % time_horizon
        
        schedule = []
        for i in range(time_horizon):
            shares = shares_per_period + (1 if i < remainder else 0)
            schedule.append((i, shares))
        
        return schedule
    
    def optimize_execution_schedule(self,
                                   total_shares: int,
                                   time_horizon: int,
                                   daily_volume: float,
                                   volatility: float,
                                   price: float) -> List[Tuple[int, int]]:
        """
        Optimize execution schedule to minimize cost + risk.
        
        Uses Almgren-Chriss optimal execution model.
        
        Args:
            total_shares: Total shares to execute
            time_horizon: Number of periods
            daily_volume: Average daily volume
            volatility: Price volatility
            price: Current price
        
        Returns:
            Optimal execution schedule
        """
        self._log(f"🎯 Optimizing execution for {total_shares:,} shares over {time_horizon} periods")
        
        # Almgren-Chriss optimal trajectory
        # Minimize: Cost + λ * Variance
        
        # Time decay parameter
        kappa = self.risk_aversion * volatility ** 2
        
        # Impact parameters (simplified)
        eta = 0.1 * volatility * price  # Temporary impact
        gamma = 0.05 * volatility * price  # Permanent impact
        
        # Optimal trajectory (exponential decay)
        tau = np.arange(time_horizon + 1)
        sinh_term = np.sinh(2 * kappa * (time_horizon - tau))
        sinh_total = np.sinh(2 * kappa * time_horizon)
        
        holdings = total_shares * sinh_term / sinh_total
        
        # Trade list (differences in holdings)
        trades = -np.diff(holdings)
        
        # Ensure non-negative and round
        trades = np.maximum(trades, 0).astype(int)
        
        # Adjust for rounding
        shortfall = total_shares - trades.sum()
        if shortfall != 0:
            trades[-1] += shortfall
        
        schedule = [(i, int(trade)) for i, trade in enumerate(trades) if trade > 0]
        
        self._log(f"✅ Optimal schedule created: {len(schedule)} periods")
        
        return schedule
    
    def _generate_default_volume_profile(self, periods: int) -> List[float]:
        """Generate U-shaped intraday volume profile."""
        # First hour: high volume
        # Middle: low volume
        # Last hour: high volume
        
        profile = []
        for i in range(periods):
            # U-shape: high at start and end
            t = i / periods
            volume = 1.0 + 0.5 * (np.cos(2 * np.pi * t) + 1)
            profile.append(volume)
        
        return profile
    
    def estimate_slippage(self,
                         order_size: int,
                         order_type: str,
                         daily_volume: float,
                         volatility: float,
                         price: float,
                         spread: float = 0.01) -> Dict[str, float]:
        """
        Estimate expected slippage for an order.
        
        Args:
            order_size: Number of shares
            order_type: 'market', 'limit', or 'vwap'
            daily_volume: Average daily volume
            volatility: Price volatility
            price: Current price
            spread: Bid-ask spread
        
        Returns:
            Slippage estimates
        """
        # Market impact
        impact = self.estimate_market_impact(order_size, daily_volume, volatility, price)
        
        # Slippage components
        slippage_components = {
            'spread_cost': spread / 2,  # Half spread
            'market_impact': impact['total_impact'],
            'timing_cost': 0
        }
        
        if order_type == 'market':
            # Market orders: pay full spread + impact
            slippage_components['spread_cost'] = spread
            slippage_components['timing_cost'] = 0
            
        elif order_type == 'limit':
            # Limit orders: save spread but risk non-execution
            slippage_components['spread_cost'] = 0
            slippage_components['timing_cost'] = volatility * price * 0.5  # Opportunity cost
            
        elif order_type == 'vwap':
            # VWAP: half spread + reduced impact
            slippage_components['spread_cost'] = spread / 2
            slippage_components['market_impact'] *= 0.7  # VWAP reduces impact
        
        total_slippage = sum(slippage_components.values())
        
        return {
            **slippage_components,
            'total_slippage': total_slippage,
            'total_slippage_bps': (total_slippage / price) * 10000,
            'order_type': order_type
        }
    
    def transaction_cost_analysis(self,
                                  execution_price: float,
                                  benchmark_price: float,
                                  shares: int,
                                  fees: float = 0) -> Dict[str, float]:
        """
        Perform Transaction Cost Analysis (TCA).
        
        Compares actual execution to benchmark.
        
        Args:
            execution_price: Actual execution price
            benchmark_price: Benchmark price (e.g., arrival price, VWAP)
            shares: Number of shares
            fees: Broker fees
        
        Returns:
            TCA metrics
        """
        # Price impact
        price_impact = execution_price - benchmark_price
        price_impact_bps = (price_impact / benchmark_price) * 10000
        
        # Total cost
        total_cost = price_impact * shares + fees
        total_cost_bps = (total_cost / (benchmark_price * shares)) * 10000
        
        return {
            'execution_price': execution_price,
            'benchmark_price': benchmark_price,
            'price_impact': price_impact,
            'price_impact_bps': price_impact_bps,
            'fees': fees,
            'total_cost': total_cost,
            'total_cost_bps': total_cost_bps,
            'shares': shares
        }
    
    def smart_order_routing(self,
                           order_size: int,
                           venues: List[Dict[str, any]]) -> List[Tuple[str, int]]:
        """
        Route order across multiple venues to minimize cost.
        
        Args:
            order_size: Total shares to execute
            venues: List of venue information [{'name': str, 'liquidity': float, 'fee': float}]
        
        Returns:
            Optimal routing [(venue_name, shares)]
        """
        self._log(f"🔀 Smart routing {order_size:,} shares across {len(venues)} venues")
        
        # Sort venues by cost (fee + liquidity impact)
        venues_sorted = sorted(venues, key=lambda v: v['fee'])
        
        routing = []
        remaining = order_size
        
        for venue in venues_sorted:
            if remaining <= 0:
                break
            
            # Allocate based on venue liquidity
            max_shares = int(venue['liquidity'])
            allocated = min(remaining, max_shares)
            
            if allocated > 0:
                routing.append((venue['name'], allocated))
                remaining -= allocated
        
        self._log(f"✅ Routed to {len(routing)} venues")
        
        return routing
    
    def generate_execution_report(self,
                                 order: Dict[str, any],
                                 schedule: List[Tuple[int, int]],
                                 estimated_costs: Dict[str, float]) -> Dict:
        """
        Generate comprehensive execution report.
        
        Args:
            order: Order details
            schedule: Execution schedule
            estimated_costs: Cost estimates
        
        Returns:
            Execution report
        """
        report = {
            'order': order,
            'schedule': {
                'total_periods': len(schedule),
                'periods': schedule,
                'average_size': np.mean([s[1] for s in schedule]) if schedule else 0
            },
            'cost_estimates': estimated_costs,
            'execution_strategy': {
                'impact_model': self.impact_model,
                'participation_rate': self.participation_rate
            }
        }
        
        return report


def main():
    """Demo the execution intelligence engine."""
    print("🎯 Execution Intelligence Engine Demo")
    print("=" * 70)
    
    # Order parameters
    total_shares = 100000
    price = 150.0
    daily_volume = 5000000
    volatility = 0.02  # 2% daily volatility
    time_horizon = 10  # 10 periods
    
    # Initialize engine
    engine = ExecutionIntelligenceEngine(verbose=True)
    
    # 1. Market Impact Estimation
    print("\n📊 Market Impact Estimation:")
    impact = engine.estimate_market_impact(total_shares, daily_volume, volatility, price)
    print(f"  Temporary Impact: ${impact['temporary_impact']:.4f} ({impact['total_impact_bps']:.1f} bps)")
    print(f"  Permanent Impact: ${impact['permanent_impact']:.4f}")
    print(f"  Total Impact: ${impact['total_impact']:.4f}")
    print(f"  Volume Participation: {impact['participation']:.2%}")
    
    # 2. VWAP Schedule
    print("\n📅 VWAP Execution Schedule:")
    vwap_schedule = engine.calculate_vwap_schedule(total_shares, time_horizon)
    print(f"  Total periods: {len(vwap_schedule)}")
    for period, shares in vwap_schedule[:5]:
        print(f"    Period {period}: {shares:,} shares")
    
    # 3. Optimal Execution
    print("\n🎯 Optimal Execution Schedule:")
    optimal_schedule = engine.optimize_execution_schedule(
        total_shares, time_horizon, daily_volume, volatility, price
    )
    
    # 4. Slippage Estimation
    print("\n💸 Slippage Estimates:")
    for order_type in ['market', 'limit', 'vwap']:
        slippage = engine.estimate_slippage(
            total_shares, order_type, daily_volume, volatility, price
        )
        print(f"  {order_type.upper()}: {slippage['total_slippage_bps']:.1f} bps")
    
    # 5. Smart Order Routing
    print("\n🔀 Smart Order Routing:")
    venues = [
        {'name': 'NYSE', 'liquidity': 50000, 'fee': 0.0005},
        {'name': 'NASDAQ', 'liquidity': 40000, 'fee': 0.0006},
        {'name': 'BATS', 'liquidity': 30000, 'fee': 0.0004},
        {'name': 'IEX', 'liquidity': 20000, 'fee': 0.0003}
    ]
    routing = engine.smart_order_routing(total_shares, venues)
    for venue, shares in routing:
        print(f"  {venue}: {shares:,} shares ({shares/total_shares:.1%})")
    
    # 6. Transaction Cost Analysis
    print("\n📋 Transaction Cost Analysis:")
    tca = engine.transaction_cost_analysis(
        execution_price=150.10,
        benchmark_price=150.00,
        shares=total_shares,
        fees=50.0
    )
    print(f"  Price Impact: {tca['price_impact_bps']:.1f} bps")
    print(f"  Total Cost: ${tca['total_cost']:,.2f} ({tca['total_cost_bps']:.1f} bps)")
    
    print("\n✅ Demo completed!")


if __name__ == "__main__":
    main()


