import os
import chromadb
from pypdf import PdfReader
from foundry_local_sdk import FoundryLocalManager, Configuration

def main():
    klasor_adi = "pdf_sources"
    
    # Klasör yoksa oluştur ve uyar
    if not os.path.exists(klasor_adi):
        os.makedirs(klasor_adi)
        print(f"📁 '{klasor_adi}' adında bir klasör oluşturuldu.")
        print("Lütfen analiz edilmesini istediğin PDF'leri bu klasörün içine atıp kodu tekrar çalıştır.")
        return

    pdf_dosyalari = [f for f in os.listdir(klasor_adi) if f.endswith('.pdf')]
    
    if not pdf_dosyalari:
        print(f"❌ '{klasor_adi}' klasöründe hiç PDF bulunamadı! Lütfen içine birkaç PDF atıp tekrar dene.")
        return
        
    print(f"📚 Toplam {len(pdf_dosyalari)} adet PDF dosyası bulundu. İşlem başlatılıyor...\n")

    print("🚀 Qwen Embedding Modeli Başlatılıyor...")
    config = Configuration(app_name="my-first-local-rag-app")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance
    
    embed_model = manager.catalog.get_model("qwen3-embedding-0.6b")
    embed_model.load()
    embed_client = embed_model.get_embedding_client()

    print("💾 Vektör veritabanı (ChromaDB) hazırlanıyor...")
    db_client = chromadb.PersistentClient(path="./vektor_db")
    
    try:
        # Eski kayıtları temizle ki üzerine yazmasın (Temiz başlangıç)
        db_client.delete_collection("pdf_notlarim")
    except:
        pass
    collection = db_client.create_collection(name="pdf_notlarim")

    toplam_chunk_sayisi = 0
    
    for dosya_adi in pdf_dosyalari:
        dosya_yolu = os.path.join(klasor_adi, dosya_adi)
        print(f"\n📖 Okunuyor: {dosya_adi}")
        
        # PDF'i Oku
        reader = PdfReader(dosya_yolu)
        tam_metin = "".join(sayfa.extract_text() + "\n" for sayfa in reader.pages)
        
        # Metni Basitçe Parçala (Her 1000 karakterde bir)
        chunk_boyutu = 1000
        chunks = [tam_metin[i:i+chunk_boyutu] for i in range(0, len(tam_metin), chunk_boyutu)]
        
        # Her bir parçayı vektöre çevirip kaydet
        for i, chunk in enumerate(chunks):
            print(f"\r  └─ İşleniyor: {i+1}/{len(chunks)}", end="", flush=True)
            vektor = embed_client.generate_embeddings(inputs=[chunk]).data[0].embedding
            
            collection.add(
                documents=[chunk],
                embeddings=[vektor],
                # Hangi parça hangi PDF'ten geldiğini metadata olarak ekliyoruz
                metadatas=[{"source": dosya_adi}], 
                # Her parça için benzersiz bir ID: ornek.pdf_chunk_0
                ids=[f"{dosya_adi}_chunk_{i}"]
            )
        toplam_chunk_sayisi += len(chunks)
        print(f"\n  ✅ {dosya_adi} veritabanına eklendi.")
    
    print(f"\n🎉 İŞLEM TAMAM! {len(pdf_dosyalari)} dosyadan toplam {toplam_chunk_sayisi} bilgi parçacığı oluşturuldu.")

if __name__ == "__main__":
    main()