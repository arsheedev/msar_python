import statsmodels.api as sm
import matplotlib.pyplot as plt

class Model:
    def __init__(
            this,
            data,
            column = (
                'CAR',
                'ROA',
                'NPF',
                'NPF_Net',
                'FDR',
                'BOPO',
                'NOM',
                'APYD_Terhadap_Aktiva_Produktif',
                'Short_Term_Mistmach'
            )
        ):
        this.data = data
        this.column = column

    def showGraph(this, index):
        model = sm.tsa.MarkovAutoregression(this.data[this.column[index - 1]], k_regimes=2, order=1)
        model_result = model.fit()
        predictions = model_result.predict()
        print(model_result.summary())

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(this.data[this.column[index - 1]], label=this.column[index - 1])
        ax.plot(model_result.filtered_marginal_probabilities[0], label='Regime')
        ax.plot(predictions, label='Predictions')
        ax.legend()
        plt.xlabel('Time (in month)')
        plt.ylabel('Value')
        plt.title('Markov Switching Autoregressive (MSAR) Model')
        plt.show()

