# FIX PROBLEM — Moteur Cadastral de Nettoyage et Purification DXF 100% ANCFCC
# دليل محرك تنظيف وتطهير ملفات DXF (« لا زايد لا ناقص »)

---

## 1. التعريف بالمجلد (Présentation)
هاد الدوسي **`FIX PROBLEM`** مجمّع فيه كود هندسي متكامل ومستقل خاص فقط بـ **تنظيف وتطهير ملفات DXF Cadastrales** وتحويلها من ملفات خام (OREGEN/Levé) إلى ملفات نهائية مطابقة 100% لمعايير الوكالة الوطنية للمحافظة العقارية والمسح العقاري والخرائطية (ANCFCC).

---

## 2. بنية المجلد (Structure du dossier)

```
FIX PROBLEM/
├── CLEAN_DXF_MASTER.py          # السكريبت الرئيسي الموحد لتشغيل جميع مراحل التنظيف
├── LANCER_NETTOYAGE.bat         # ملف تشغيل سريع بنقرة واحدة (Double-clic)
│
├── tools/                       # الأدوات الذرية الخمس (5 Outils Atomiques)
│   ├── tool_1_bornes.py         # Outil 1 : البورنات، المتجهات الخارجية، وخطوط الوصل (Rappels)
│   ├── tool_2_riverains.py      # Outil 2 : المجاورون، وحذف الطفيليات (Mosquée, Plle 1370...)
│   ├── tool_3_lineaires.py      # Outil 3 : التفاصيل الخطية (مسالك، شعاب، أودية) والتمركز
│   ├── tool_4_cartouche.py      # Outil 4 : كادر ST25، حساب المساحة بـ Gauss، والشبكة 100m
│   └── tool_5_anti_collision.py # Outil 5 : محرك منع التداخل الدقيق (Micro-SAT Anti-collision)
│
├── Coffres-Forts Expert (الخزائن المرجعية المعتمدة)
│   ├── cadastre_expert_bornes_vault.json
│   ├── cadastre_expert_riverains_vault.json
│   ├── cadastre_expert_lineaires_vault.json
│   └── cadastre_expert_cartouche_vault.json
│
├── diff_engine/                 # محرك المقارنة البصرية والرياضية وحساب الدقة
│   ├── visual_diff.py
│   ├── diff_report.py
│   └── render_diff_png.py
│
├── Moteurs de Base (محركات الحساب الهندسي)
│   ├── score_all.py             # قاضي الجودة الصارم ومقارن الكيانات
│   ├── standards.py             # معايير REJCAD وقفل الدقة السنتيمترية
│   ├── linear_centering.py      # خوارزميات تمركز النصوص على المنحنيات
│   ├── text_anti_collision.py   # خوارزميات تفادي تصادم النصوص
│   ├── geometry_clipping.py     # قص الخطوط عند حدود الكادر
│   └── sector_riverain_engine.py# محرك ربط المخططات القطاعية
```

---

## 3. طريقة الاستعمال (Mode d'Emploi)

### الطريقة الأولى: بنقرة واحدة (Double-clic)
- كليكي دوبل كليك على **`LANCER_NETTOYAGE.bat`**
- اختار:
  - رقم `1` لتنظيف ملف DXF واحد.
  - رقم `2` لتنظيف مجلد كامل دفعة واحدة.
- أو يمكنك جر ملف DXF بالماوس (Drag & Drop) وحطه فوق `LANCER_NETTOYAGE.bat`.

### الطريقة الثانية: عبر سطر الأوامر (Command Line)

#### 1. تنظيف ملف DXF واحد (Fichier Unique) :
```powershell
python CLEAN_DXF_MASTER.py --input "C:\chemin\vers\P20-PLAN.dxf" --output "C:\chemin\vers\P20_PROPRE.dxf"
```

#### 2. تنظيف مجلد كامل (Dossier Entier) :
```powershell
python CLEAN_DXF_MASTER.py --batch-dir "C:\Users\Admin\Downloads\Sous_Zone 01\IGHANOUYINE\OREGEN\..." --output-dir "C:\Users\Admin\Desktop\DXF_PROPRES"
```

---

## 4. الضمانات الهندسية (« لا زايد لا ناقص »)
1. **Zéro Parasite** : إزالة كل النصوص الغريبة القادمة عشوائياً من القطاعات المجاورة (`Mosquée`، `Plle 1370`، `Plle 19`، إلخ).
2. **Zéro Dépassement** : نقطة الارتكاز `attachment_point = 5` (MiddleCenter) تضمن بقاء النصوص داخل الكادر ST25 وعدم خروجها عن حدوده.
3. **Zéro Collision** : الكارتوش منظم في خلايا هندسية مضبوطة وتفادي تراكم الكلمات فوق بعضها.
