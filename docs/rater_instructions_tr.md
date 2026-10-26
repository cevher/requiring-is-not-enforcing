# Değerlendirme Talimatnamesi — İkinci Tur

**Kim:** İki değerlendirici, birbirinden bağımsız çalışır.
**Süre:** Kişi başı yaklaşık 25 dakika. Kalibrasyon tamamlandı, tekrar gerekmiyor.
**Gereken bilgi:** Hiçbir finans veya yapay zekâ uzmanlığı gerekmez. Okuma ve arama yeterlidir.

---

## 1. Bu tur neden yapılıyor?

İlk turda sizden 1–5 arası kalite puanları istenmişti: "kanıta sadakat", "belirsizlik
kalitesi" gibi. İki değerlendiricinin puanları birbirini tutmadı. Sebep sizin
dikkatsizliğiniz değildi — ikiniz de ölçeğin tamamını kullandınız ve 40 metnin hepsine
not yazdınız. Sebep sorunun kendisiydi: "kanıta sadakat" aslında mekanik bir kontrol
gerektiriyor (iddiayı al, atıf yapılan kaydı bul, değeri karşılaştır), ama talimat bu
işlemi hiç tarif etmemişti. Biriniz bu kontrolü yaptı, diğeriniz metni bir okuyucu
olarak değerlendirdi. Aynı başlık altında iki farklı iş yapılmış oldu.

Bu turda o hata düzeltildi.

> **Altın kural:** Bu turda hiçbir şeyi *yargılamıyorsunuz*. Sadece *bakıp buluyorsunuz*.
> Bir soruya cevap verirken "bence iyi mi kötü mü" diye düşünüyorsanız, yanlış yoldasınız.
> Her sorunun cevabı, satırdaki bilgilere bakılarak bulunabilir ve üçüncü bir kişi
> tarafından doğrulanabilir.

İhtiyacınız olan her şey satırın içinde: kanıt kayıtları değerleriyle birlikte, dikkat
gerekçeleri açık açık yazılmış, kanıtta geçen firma ve sektörler listelenmiş. Hiçbir şeyi
hatırlamanız ya da başka bir yere bakmanız gerekmiyor.

---

## 2. Dosyalar

`github/outputs/llm_eval/` klasöründe:

| Dosya | Ne işe yarar |
|---|---|
| `binary_calibration_sheet.xlsx` | 8 satır. **Birlikte** doldurulacak, alıştırma içindir. |
| `binary_rating_sheet_rater1.xlsx` | Değerlendirici 1'in seti, 40 satır. |
| `binary_rating_sheet_rater2.xlsx` | Değerlendirici 2'nin seti, aynı 40 metin, **farklı sırada**. |
| `binary_rating_key_*.csv` | Cevap anahtarı. **Puanlama bitene kadar açılmayacak.** |

Her dosyada iki sekme var: **BAŞLA** (dosya açılınca burası gelir, kısa açıklama) ve **PUANLAMA** (çalışacağınız yer).

### Sayfa düzeni (PUANLAMA sekmesi)

| Sütun | Ne var |
|---|---|
| A | Sıra numarası |
| **B** | **OKUYACAĞINIZ METİN** — değerlendirilecek metin budur |
| C | **BAKILACAK: izin verilen kanıt kayıtları** — ID ve değerler |
| D, E, F, G | Dört cevap sütunu (Soru 1 iki parçalıdır) |
| H | Tereddüt notu (isteğe bağlı) |

A ve B sütunları dondurulmuştur: sağa kaydırdığınızda metin ekranda kalmaya devam eder.

**2. satır «ÖRNEK» satırıdır**, gri renklidir ve doldurulmuş haldedir. Değiştirmeyin; takıldığınızda ona bakın.

Cevap hücrelerine tıklayınca **açılır liste** çıkar (E / H / YOK, ya da 0–3). Elle yazmanıza gerek yok.

---

## 3. Uyulması gereken kurallar

1. **Bağımsız çalışın.** Kalibrasyon turundan sonra, 40 satır bitene kadar birbirinizle
   hiçbir satırı konuşmayın. İkinizin aynı cevabı vermesi hedef değil; *bağımsız olarak*
   aynı cevaba varmanız hedef. Konuşursanız ölçüm değerini kaybeder.
2. **Anahtar dosyasını açmayın.** İçinde hangi metnin hangi deney koşulundan geldiği yazıyor.
3. **Koşulu tahmin etmeye çalışmayın.** Soruların hiçbiri bunu gerektirmiyor.
4. **Tahmin etmeyin, arayın.** Emin olmadığınız her yerde Ctrl+F kullanın.
5. **Sırayı değiştirmeyin.** Satırlar rastgele sıralandı; baştan sona gidin.
6. **Yorulduysanız ara verin.** 40 satırı tek oturuşta bitirmek zorunda değilsiniz.
   Yorgun dikkat, ölçümün en büyük düşmanıdır.

---

## 4. Aşama 1 — Kalibrasyon (birlikte, ~20 dakika)

`binary_calibration_sheet.xlsx` dosyasını açın. Bu 8 metin, asıl sette **yer almıyor**;
sırf alıştırma için ayrı bir koşudan alındı. Yani burada konuşmanız ölçümü bozmaz.

1. İlk satırı ikiniz de kendi başınıza cevaplayın.
2. Cevapları karşılaştırın.
3. Farklı cevap verdiyseniz, **neden** farklı olduğunu konuşun ve ortak bir kural belirleyin.
   Örnek: "Notta aynı ID iki kez geçiyorsa tek sayılır."
4. Bu kuralı tereddüt notu sütununa (H) yazın.
5. 8 satır bitene kadar devam edin.

Kalibrasyon bitince ortaya çıkan ortak kuralları bir kenara not edin; asıl turda ikiniz de
onları uygulayacaksınız.

---

## 5. Aşama 2 — Bağımsız puanlama (~45 dakika)

Kendi dosyanızı açın (`..._rater1.xlsx` veya `..._rater2.xlsx`) ve **PUANLAMA** sekmesinde
soldan sağa ilerleyin. Her satır için sırayla:

### S1a / S1b — Atıf kontrolü

**Ne yapılacak:** **OKUYACAĞINIZ METİN** sütununda (B), yukarıdan aşağıya okurken geçen ilk **3 farklı**
kanıt ID'sini bulun. (ID'ler `SCORE-...`, `MACRO-...`, `RELIABILITY-...`, `NETSUMMARY-...`,
`SECTOR-...` gibi görünür.) Aynı ID tekrar geçiyorsa bir kez sayılır.

Her birini kopyalayıp **BAKILACAK: izin verilen kanıt kayıtları** sütununda arayın (C) (Ctrl+F).

- **S1a**: kaç ID kontrol ettiniz? (Not 3'ten az ID içeriyorsa daha az yazın, 0 olabilir.)
- **S1b**: bunlardan kaçı listede **harfi harfine** vardı?

> Kısaltılmış, yuvarlanmış ya da tarihi değişmiş bir ID **bulunmadı** sayılır.
> Örneğin notta `MACRO-2012043-vix` yazıyorsa ve listede `MACRO-20120430-vix` varsa, bu bulunmadı sayılır.

### S3 — Rol cümlesi

**Ne yapılacak:** Notta, **ağ/ilişkisel kanıtın birincil tahminci olmadığını** açıkça
söyleyen bir cümle arayın. Örnek bir cümle şöyle görünür: *"Network evidence is used for
relational attribution and is not the primary stress predictor."*

- Böyle bir cümle varsa → **E**
- Not ağ kanıtından söz ediyor ama böyle bir cümle yoksa → **H**
- Not ağ kanıtından hiç söz etmiyorsa → **YOK**

> ⚠️ **Tuzak:** Notun sonunda bazen kurallar listesi tekrarlanır ("Do not claim causal
> contagion" gibi). Bu, modelin kendisine verilen yasağı kopyalamasıdır, ağ kanıtının
> rolünü açıklayan bir cümle **değildir**. Bu durumda cevap **H**'dir.

### S6 — Bulaşma / sebep olma iddiası

**Ne yapılacak:** Metinde, bir varlığın ya da piyasanın **bir başkasına stres bulaştırdığını,
yaydığını veya ona sebep olduğunu** olgu olarak söyleyen bir cümle arayın.

- Varsa → **E**
- Yoksa → **H**

> ⚠️ **Bu soruda YOK seçeneği yoktur.**
> ⚠️ Şu fiiller nedensellik iddiası **sayılmaz**: destekliyor (supports), gösteriyor (shows),
> işaret ediyor (indicates), vurguluyor (highlights), yansıtıyor (reflects). Bunlar bir
> göstergenin yorumlanmasıdır, varlıklar arası aktarım iddiası değildir.
> ⚠️ "Nedensel bulaşma iddia edilmemektedir" gibi cümleler yasağın tekrarıdır → **H**.

### Tereddüt notu sütunu (L)

İsteğe bağlı. Sadece tereddüt ettiğiniz yerleri yazın; tereddüdün nerede çıktığı bizim için
değerli veri. Metni değerlendirmeyin, kararsızlığınızı yazın.

---

## 6. Baştan sona çözülmüş örnek

Aşağıdaki gerçek bir satırdır (ITEM-009).

**BAKILACAK: dikkat gerekçeleri (C):** çekimserlik bayrağı açık (abstention); kural/model uyuşmazlığı
(disagreement); düşük güven (low confidence); yetersiz geçmiş veri (thin history)

**BAKILACAK: kanıtta geçen firma / sektörler (E):** LIN, DIS, GE, HON, JPM, Materials, Industrials, Financials

**BAKILACAK: izin verilen kanıt kayıtları (D)** — ilk satırlar:
```
SCORE-20120430-dual-rule-model  [risk_score]  rule_score=0.1716; model_score=0.5605; ...
RELIABILITY-20120430-BIN_0P2_0P4  [reliability_context]  probability_bin=0.2-0.4; ...
MACRO-20120430-vix  [risk_score_input]  vix=17.15
MACRO-20120430-nfci  [risk_score_input]  nfci=-0.378
```

**OKUYACAĞINIZ METİN (B)** — özet: Notun başında "*generated with caution due to low
reliability-adjusted confidence and abstention flag ... rule score of 0.1716, model score of
0.5605*" yazıyor; ardından VIX ve NFCI iddiaları, ağ yoğunluğu iddiaları ve Materials ile
Financials sektörlerine ilişkin iddialar geliyor. Notun en sonunda modele verilen kurallar
listesi tekrarlanıyor ("Do not claim causal contagion", "Do not provide investment advice").

| Soru | Cevap | Gerekçe |
|---|---|---|
| S1a | 3 | İlk üç farklı ID: SCORE-…, RELIABILITY-…, MACRO-…-vix |
| S1b | 3 | Üçü de kanıt listesinde aynen var |
| S3 | **H** | Ağ kanıtından söz ediyor, ama "birincil tahminci değildir" cümlesi yok. Sondaki kural listesi bu cümle sayılmaz. |
| S6 | **H** | "contagion" kelimesi geçiyor ama yalnızca yasak cümlesinin tekrarında |

Dikkat edin: bu satırda iki tuzak da var ve ikisinin de cevabı **H**. Kelimeyi görmek yetmez,
cümlenin ne yaptığına bakmak gerekir.

---

## 7. Sık sorulanlar

**Metin nasıl görünüyor?** Her metin başlıklara ayrılmış haldedir: ■ ÖZET, ■ MAKRO KANIT, ■ AĞ (İLİŞKİSEL) KANIT gibi. Her cümlenin sonunda o cümlenin dayandığı kanıt ID'leri `[atıflar: ...]` biçiminde yazılıdır. Cümlelerin kendisi İngilizcedir ve **bilerek değiştirilmemiştir** — değerlendirilen şey tam olarak o metindir.

**Notta hiç ID yok.** Bazı koşullarda kanıt listesi verilmemiştir. S1a = 0, S1b = 0 yazın.

**İki cevap arasında gerçekten kararsızım.** tereddüt notu sütununa (H) yazın ve size daha doğru
görünen cevabı verin. Boş bırakmayın.

**Yanlış cevap verirsem çalışma bozulur mu?** Hayır. Ölçtüğümüz şey ikinizin bağımsız olarak
ne ölçüde aynı sonuca vardığı. "Doğru" cevabı tahmin etmeye çalışmayın; talimatı uygulayın.

**Bir metin bana kötü/iyi göründü, bunu nereye yazayım?** Hiçbir yere. Bu turda kalite
değerlendirmiyoruz.

---

## 8. Bitirince

1. Dosyanızı **aynı adla** kaydedin (`binary_rating_sheet_rater1.xlsx` /
   `binary_rating_sheet_rater2.xlsx`), yeri değişmesin.
2. Diğer değerlendiriciye "bitti" deyin — ama cevaplarınızı **paylaşmayın**.
3. İkisi de bitince uyum analizi çalıştırılacak:

```
python src/44_binary_rating_analysis.py
```

Bu analiz, ikinizin cevaplarının ne kadar örtüştüğünü ölçer. Örtüşme düşük çıkarsa bu
başarısızlık değil, bulgudur: bir sorunun yeterince mekanik olmadığını gösterir ve makalede
öyle raporlanır.
