# Penjelasan kode MSAR Model Hemilton

Kode tersebut merupakan implementasi dari model Markov Switching Autoregressive (MSAR) menggunakan pustaka `statsmodels` pada bahasa pemrograman Python. Berikut adalah penjelasan rinci dari setiap baris kode:

1. `import pandas as pd`: Ini adalah pernyataan impor yang mengimpor pustaka pandas dengan alias `pd`. Pandas adalah pustaka populer untuk memanipulasi dan menganalisis data.

2. `import matplotlib.pyplot as plt`: Ini adalah pernyataan impor yang mengimpor modul `pyplot` dari pustaka matplotlib dengan alias `plt`. Matplotlib adalah pustaka visualisasi data yang digunakan di sini untuk membuat plot.

3. `from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression`: Ini adalah pernyataan impor yang mengimpor kelas `MarkovAutoregression` dari modul `markov_autoregression` dalam pustaka `statsmodels.tsa.regime_switching`. Kelas ini digunakan untuk memodelkan MSAR.

4. `data = pd.read_csv('data.csv')`: Ini adalah langkah untuk membaca data dari file CSV yang disimpan dalam variabel `data`. Data ini kemudian digunakan dalam pemodelan MSAR.

5. `values = data['CAR'].values`: Ini mengambil kolom 'CAR' dari dataset yang telah dibaca dan menyimpan nilainya dalam variabel `values`. Kolom 'CAR' diasumsikan berisi data yang akan dimodelkan.

6. `model = MarkovAutoregression(values, k_regimes=2, order=1)`: Membuat objek model MSAR dengan memanggil kelas `MarkovAutoregression`. Argumen pertama `values` adalah data yang akan dimodelkan. Argumen `k_regimes` adalah jumlah rezim dalam model (dalam kasus ini 2 rezim), dan `order` adalah urutan model autoregresi (dalam kasus ini urutan 1).

7. `result = model.fit()`: Menggunakan metode `fit()` pada objek model untuk memperoleh hasil pemodelan MSAR. Metode ini akan memperkirakan parameter model menggunakan data yang diberikan.

8. `regime_probs = result.filtered_marginal_probabilities[0]`: Menyimpan probabilitas rezim hasil pemodelan dalam variabel `regime_probs`. Probabilitas ini merupakan hasil dari proses filtering pada model MSAR dan dalam kasus ini mengambil probabilitas rezim pertama (indeks 0).

9. `plt.figure(figsize=(10, 6))`: Membuat objek gambar baru dengan ukuran 10x6 inci menggunakan fungsi `figure()` dari pustaka `pyplot`.

10. `plt.plot(values, label='Data', color='blue')`: Membuat plot garis dari data awal (`values`) dengan label "Data" dan warna biru.

11. `plt.plot(regime_probs, label='Regime 0', color='red')`: Membuat plot garis dari probabilitas rezim (`regime_probs`) dengan label "Regime 0" dan warna merah.

12. `plt.xlabel('Time')`: Memberikan label sumbu x pada plot sebagai "Time".

13. `plt.ylabel('Value')`: Memberikan label sumbu y pada plot sebagai "Value".

14. `plt.title('Markov Switching Autoregressive Model')`: Memberikan judul plot sebagai

# Penjelasan kode MSAR Model Markov Chain Monte Carlo (mcmc.py)

Kode di atas merupakan implementasi pemodelan Markov Switching Autoregressive (MSAR) menggunakan pustaka PyMC3 pada bahasa pemrograman Python. Berikut adalah penjelasan rinci dari setiap baris kode:

1. `import numpy as np`: Ini adalah pernyataan impor yang mengimpor pustaka numpy dengan alias `np`. NumPy adalah pustaka yang digunakan untuk komputasi numerik.

2. `import pymc3 as pm`: Ini adalah pernyataan impor yang mengimpor pustaka PyMC3 dengan alias `pm`. PyMC3 adalah pustaka untuk pemodelan probabilistik bayesian menggunakan metode Monte Carlo Markov Chain (MCMC).

3. `import pandas as pd`: Ini adalah pernyataan impor yang mengimpor pustaka pandas dengan alias `pd`. Pandas adalah pustaka populer untuk memanipulasi dan menganalisis data.

4. `import matplotlib.pyplot as plt`: Ini adalah pernyataan impor yang mengimpor modul `pyplot` dari pustaka matplotlib dengan alias `plt`. Matplotlib adalah pustaka visualisasi data yang digunakan di sini untuk membuat plot.

5. `data = pd.read_csv('data.csv')`: Ini adalah langkah untuk membaca data dari file CSV yang disimpan dalam variabel `data`. Data ini kemudian digunakan dalam pemodelan MSAR.

6. `observations = data['Short_Term_Mistmach'].dropna().values`: Ini mengambil kolom 'Short_Term_Mistmach' dari dataset yang telah dibaca dan menyimpan nilai-nilainya dalam variabel `observations`. Nilai-nilai yang tidak valid (NaN) dihapus menggunakan metode `dropna()`.

7. Model specification:
   a. `with pm.Model() as model:`: Ini membuat objek model PyMC3 baru dan menjalankan blok pernyataan yang berikutnya dalam konteks model ini.
   b. `p = pm.Beta("p", alpha=1, beta=1, shape=2)`: Ini mendefinisikan prior untuk probabilitas transisi antara rezim-rezim dalam MSAR. Variabel `p` adalah variabel acak yang mengikuti distribusi Beta dengan parameter alpha=1 dan beta=1. Shape=2 menunjukkan bahwa ada 2 rezim.
   c. `mu = pm.Normal("mu", mu=0, sigma=10, shape=2)`: Ini mendefinisikan prior untuk mean (rata-rata) dari masing-masing rezim. Variabel `mu` adalah variabel acak yang mengikuti distribusi normal dengan mean 0 dan standard deviation (deviasi standar) 10. Shape=2 menunjukkan bahwa ada 2 rezim.
   d. `sigma = pm.HalfNormal("sigma", sigma=10, shape=2)`: Ini mendefinisikan prior untuk standar deviasi dari masing-masing rezim. Variabel `sigma` adalah variabel acak yang mengikuti distribusi Half-Normal dengan standard deviation 10. Shape=2 menunjukkan bahwa ada 2 rezim.
   e. `states = pm.Categorical("states", p=p, shape=len(observations))`: Ini mendefinisikan variabel tersembunyi (hidden states) dalam MSAR. Variabel `states` adalah variabel acak yang mengikuti distribusi kategorikal dengan probabilitas transisi `p` dan jumlah observasi yang sama dengan panjang `observations`.
   f. `obs = pm.Normal("obs", mu=mu[states], sigma=sigma[states], observed=observations)`: Ini mendefinisikan observasi dalam MSAR. Variabel `obs`adalah variabel acak yang mengikuti distribusi normal dengan mean`mu`yang tergantung pada`states`dan standar deviasi`sigma`yang juga tergantung pada`states`. Nilai yang diamati (`observations`) diberikan sebagai argumen `observed`.

8. `trace = pm.sample(2000, tune=1000, chains=2)`: Ini melakukan sampling dari posterior distribution menggunakan metode MCMC dengan 2000 iterasi (samples) setelah tahap tuning sebanyak 1000 iterasi. Chains=2 menunjukkan bahwa ada dua rantai Markov dalam MCMC.

9. `n_predictions = 100`: Ini adalah jumlah prediksi yang akan dihasilkan.

10. `posterior_states = pm.sample_posterior_predictive(trace, model=model, samples=n_predictions)`: Ini menghasilkan prediksi menggunakan distribusi posterior dari hidden states (`states`) berdasarkan hasil sampling `trace` dengan menggunakan metode `sample_posterior_predictive()` pada objek model.

11. `predicted_observations = posterior_states["obs"].mean(axis=0)`: Ini mengambil nilai rata-rata prediksi observasi (`obs`) dari `posterior_states` yang dihasilkan sebelumnya.

12. Plot original observations and predicted values:
    a. `plt.figure()`: Membuat objek gambar baru menggunakan fungsi `figure()` dari pustaka `pyplot`.
    b. `plt.plot(np.arange(len(observations)), observations, label="Data Asli")`: Membuat plot garis dari observasi asli (`observations`) dengan label "Data Asli".
    c. `plt.plot(predicted_observations, label="Prediksi")`: Membuat plot garis dari nilai prediksi (`predicted_observations`) dengan label "Prediksi".
    d. `plt.title('MSAR Grafik Short_Term_Mistmach')`: Memberikan judul plot sebagai "MSAR Grafik Short_Term_Mistmach".
    e. `plt.legend()`: Menampilkan legenda pada plot.
    f. `plt.show()`: Menampilkan plot.

# Penjelasan kode MSAR Model Markov Chain Monte Carlo (msar.py)

Kode di atas merupakan implementasi pemodelan Markov Switching Autoregressive (MSAR) menggunakan pustaka PyMC3 pada bahasa pemrograman Python. Berikut adalah penjelasan rinci dari setiap baris kode:

1. `import pandas as pd`: Ini adalah pernyataan impor yang mengimpor pustaka pandas dengan alias `pd`. Pandas adalah pustaka populer untuk memanipulasi dan menganalisis data.

2. `import pymc3 as pm`: Ini adalah pernyataan impor yang mengimpor pustaka PyMC3 dengan alias `pm`. PyMC3 adalah pustaka untuk pemodelan probabilistik bayesian menggunakan metode Monte Carlo Markov Chain (MCMC).

3. `import numpy as np`: Ini adalah pernyataan impor yang mengimpor pustaka numpy dengan alias `np`. NumPy adalah pustaka yang digunakan untuk komputasi numerik.

4. `import matplotlib.pyplot as plt`: Ini adalah pernyataan impor yang mengimpor modul `pyplot` dari pustaka matplotlib dengan alias `plt`. Matplotlib adalah pustaka visualisasi data yang digunakan di sini untuk membuat plot.

5. `data = pd.read_csv('data.csv')`: Ini adalah langkah untuk membaca data dari file CSV yang disimpan dalam variabel `data`. Data ini kemudian digunakan dalam pemodelan MSAR.

6. `observations = data['Short_Term_Mistmach'].dropna().values`: Ini mengambil kolom 'Short_Term_Mistmach' dari dataset yang telah dibaca dan menyimpan nilai-nilainya dalam variabel `observations`. Nilai-nilai yang tidak valid (NaN) dihapus menggunakan metode `dropna()`.

7. Model specification:
   a. `with pm.Model() as model:`: Ini membuat objek model PyMC3 baru dan menjalankan blok pernyataan yang berikutnya dalam konteks model ini.
   b. `sigma1 = pm.Exponential('sigma1', 1.0)`: Ini mendefinisikan prior untuk parameter sigma1 dalam MSAR. Variabel `sigma1` adalah variabel acak yang mengikuti distribusi eksponensial dengan parameter 1.0.
   c. `sigma2 = pm.Exponential('sigma2', 1.0)`: Ini mendefinisikan prior untuk parameter sigma2 dalam MSAR. Variabel `sigma2` adalah variabel acak yang mengikuti distribusi eksponensial dengan parameter 1.0.
   d. `p = pm.Dirichlet('p', a=np.array([1, 1]))`: Ini mendefinisikan prior untuk probabilitas transisi antara rezim-rezim dalam MSAR. Variabel `p` adalah variabel acak yang mengikuti distribusi Dirichlet dengan parameter alpha sebesar [1, 1]. Distribusi Dirichlet digunakan karena memodelkan distribusi probabilitas yang mengikuti batasan non-negatif dan jumlah total probabilitas yang sama dengan 1.
   e. `state = pm.Categorical('state', p=p, shape=len(observations))`: Ini mendefinisikan variabel tersembunyi (hidden states) dalam MSAR. Variabel `state` adalah variabel acak yang mengikuti distribusi kategorikal dengan probabilitas transisi `p` dan jumlah observasi yang sama dengan panjang `observations`.
   f. Definisi komponen autoregresi:
   - `coeff1 = pm.Normal('coeff1', mu=0, sd=10)`: Ini mendefinisikan prior untuk koefisien coeff1 dalam model autoregresi. Variabel `coeff1`adalah variabel acak yang mengikuti distribusi normal dengan mean 0 dan standard deviation 10. -`coeff2 = pm.Normal('coeff2', mu=0, sd=10)`: Ini mendefinisikan prior untuk koefisien coeff2 dalam model autoregresi. Variabel `coeff2`adalah variabel acak yang mengikuti distribusi normal dengan mean 0 dan standard deviation 10. -`rho1 = pm.Normal('rho1', mu=0, sd=1)`: Ini mendefinisikan prior untuk koefisien autoregresi rho1 dalam model autoregresi. Variabel `rho1`adalah variabel acak yang mengikuti distribusi normal dengan mean 0 dan standard deviation 1. -`rho2 = pm.Normal('rho2', mu=0, sd=1)`: Ini mendefinisikan prior untuk koefisien autoregresi rho2 dalam model autoregresi. Variabel `rho2`adalah variabel acak yang mengikuti distribusi normal dengan mean 0 dan standard deviation 1. -`ar1 = pm.AR('ar1', rho=rho1, shape=len(observations))`: Ini mendefinisikan komponen autoregresi ar1 dalam model autoregresi. Variabel `ar1`adalah variabel acak yang mengikuti distribusi autoregresi dengan koefisien autoregresi`rho1`dan jumlah observasi yang sama dengan panjang`observations`.
   - `ar2 = pm.AR('ar2', rho=rho2, shape=len(observations))`: Ini mendefinisikan komponen autoregresi ar2 dalam model autoregresi. Variabel `ar2`adalah variabel acak yang mengikuti distribusi autoregresi dengan koefisien autoregresi`rho2`dan jumlah observasi yang sama dengan panjang`observations`.
   - `mu = coeff1 _ ar1 + coeff2 _ ar2`: Ini mendefinisikan nilai rata-rata observasi dalam MSAR sebagai kombinasi linier dari komponen autoregresi (`ar1`dan`ar2`) dengan koefisien `coeff1`dan`coeff2`.

g. Model likelihood: - `obs_sigma = pm.math.switch(state, sigma1, sigma2)`: Ini mendefinisikan standar deviasi observasi (`obs_sigma`) dalam MSAR berdasarkan nilai rezim tersembunyi (`state`). Jika `state` adalah 0, maka `obs_sigma` adalah `sigma1`. Jika `state` adalah 1, maka `obs_sigma` adalah `sigma2`. - `obs = pm.Normal('obs', mu=mu, sd=obs_sigma, observed=observations)`: Ini mendefinisikan likelihood untuk observasi dalam MSAR. Variabel `obs` adalah variabel acak yang mengikuti distribusi normal dengan mean `mu` (rata-rata) dan standar deviasi `obs_sigma` (yang bergantung pada `state`). Nilai yang diamati (`observations`) diberikan sebagai argumen `observed`.

8. `trace = pm.sample(2000, tune=1000, chains=2)`: Ini melakukan sampling dari posterior distribution menggunakan metode MCMC dengan 2000 iterasi (samples) setelah tahap
