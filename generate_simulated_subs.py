"""
Generating Statement Sense Simulated Subscriptions

Outputs Simulated Subscriptions CSV
- Simulated Subscriptions have random and realistic values for features:
    - merchant
    - usage_intensity    
    - cost_efficiency*    
    - n_charges
    - trial_prob  
    - monthly_charge_jmd 

***Suggestion: Function that checks the existence of other related apps on the phone to assess the cost-efficiency of sub. 
                For Example: Gym/health related apps would highlight gym subscription is actually HIGH VALUE

***Concerns: 
    - Not sure if I should let user priority be the most dominant weighting factor because
        will the model never recommend a cancellation again. SHould user priority be seen as a
        minor tie breaker or the dominant weighting factor the trumps every other feature of the value score?
        *Possible solution: Override attached to merchant name embedded in Assigning Value Risk

***TODO: 
    - renewal_consistency deferred — requires billing cadence detection
        Biannual/quarterly charges produce different n_charges counts over the
        same duration_months, making a naive actual/expected ratio unreliable.


"""

import numpy as np
import pandas as pd

#declaring CONSTANT for JMD-USD Conversion
JMD_RATE = 156.0

def generate_subscription_data(n_samples: int = 2000, random_state: int = 42) -> pd.DataFrame:
    np.random.seed(random_state)

    data = {
        #randomly assigns one real-world variation to merchant
        "merchant": np.random.choice(["NFLX", "SPOT", "ADBE", "AMZN"], n_samples),

        #number of charges seen in subscription. Poisson gives realistic spread for 6 months.
        #min 1 charge and max 20 charges
        "n_charges": np.random.poisson(6, n_samples).clip(1, 20),

        #avg monthly charge for sub
        "avg_monthly_usd": np.random.uniform(5, 30, n_samples),

        #0-never, 1-daily usage 
        "usage_intensity": np.random.uniform(0, 1, n_samples),

        #Beta(5,2) skews high trial probability just thinking in the Ja context
        "trial_prob": np.random.beta(5, 2, n_samples),

        #actual length of time the subscription has been active for in MONTHS.
        #treating as either really long for 48 months or newly subscribed at 1
        "duration_months": np.random.exponential(12, n_samples).clip(1, 48),

        #weight reflects how much user has values sub, set by user prompt
        #0.5 is low, 1.0 is neutral, 1.5 is high
        #creating discrete weighting factor with a neutral baseline
        "user_priority": np.random.choice([0.5, 1.0, 1.5], n_samples)
    }
    df = pd.DataFrame(data)

    #Conversion of USD to JMD 
    df["monthly_charge_jmd"] = df["avg_monthly_usd"] * JMD_RATE


    #normalize duration to 0-1
    df["duration_score"] = (df["duration_months"] / 48).clip(0, 1)

  
    #higher usage + lower monthly cost = better cost efficiency
    df["cost_efficiency"] = (
        df["usage_intensity"] / (df["avg_monthly_usd"] / 15)
    ).clip(0, 1)

    #weighted heuristic value score WITH user priority as a tiebreaker weight
    df["value_score"] = (
    0.30 * df["usage_intensity"] +
    0.25 * df["cost_efficiency"] +
    0.25 * df["duration_score"] +
    0.20 * (df["user_priority"] / 1.5) -
    0.15 * df["trial_prob"]  #explicit trial penalty, separate from weights
    ).clip(0, 1) * 100

    

    # Assign risk labels
    df["risk_label"] = pd.cut(
        df["value_score"],
        bins=[-np.inf, 40, 70, np.inf],
        labels=["High", "Medium", "Low"]
    )

    return df


if __name__ == "__main__":
    df = generate_subscription_data()
    df.to_csv("simulated_subscriptions.csv", index=False)

    print("Simulated subscription dataset created successfully.")
    print(f"\nRisk label distribution:\n{df['risk_label'].value_counts()}")
    print(f"\nMean value score: {df['value_score'].mean():.1f}")
    print(f"\nSample Records Generated: \n {df.head()}")




