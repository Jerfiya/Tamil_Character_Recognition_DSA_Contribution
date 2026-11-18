# Dataset audit

- Dataset root (found by scan): `C:\Users\Dell\Downloads\project_kl\Tamil Letters Combined`
- Classes with images: **247** — per category: {'ஆய்தம': 1, 'உயிர் எழுத்துக்கள': 12, 'உயிர்மெய் எழுத்துக்கள': 216, 'மெய் எழுத்துக்கள்': 18}
- Files seen: 14463; usable after cleaning: **14205**
- Images per class (raw): min 4 / median 47 / max 98
- Images per class (after dedupe): min 4 / median 46 / max 96
- Small classes (<15 images after dedupe): {'ணே': 4, 'ணெ': 4, 'ழா': 4, 'ழ': 4, 'ணு': 5, 'ணூ': 5, 'ழி': 6, 'ழீ': 6}
- Empty class folders (skipped, enrollable later): none
- Nested class folders: none
- Classes merged from several folders: none
- Zip archives: none (0 images extracted)
- Formats: {'.png': 14463}
- Corrupt files (skipped): 0 none
- Cross-class identical images removed as label conflicts: 0
- Within-class near-duplicates removed (pHash): 258
- Near-duplicate count by Hamming threshold: {'0': 195, '2': 234, '4': 258, '6': 309, '8': 461, '10': 910}
- Source groups: {'other': 11013, 'numeric': 1324, 'screenshot': 781, 'img_date': 374, 'whatsapp': 278, 'resized': 248, 'scanned': 161, 'pxl': 26}; origins: {'ds1': 10790, 'ds2': 3415}; distinct writer keys: 171
- Image size: width {'min': 48, 'median': 48.0, 'max': 48}, height {'min': 48, 'median': 48.0, 'max': 48}, distinct sizes 1
- Polarity: {'dark_ink_on_light': 14180, 'light_ink_on_dark': 25}

## Alphabet coverage (18 x 12 grid)

- Missing compounds: தௌ, லௌ, ளௌ
- Missing vowels: none; missing consonants: none; missing aytham: none
- **Possible spelling variants (kept as distinct classes, NOT merged):** தெள, லெள, ளெள — possible variants of தௌ, லௌ, ளௌ
- Unrecognised labels: none
- Enrollable later: தௌ, லௌ, ளௌ

## Episode eligibility (classes with at least K+Q images)

| requirement | classes |
|---|---|
| K+Q>=2 | 247 |
| K+Q>=6 | 241 |
| K+Q>=10 | 239 |
| K+Q>=15 | 239 |
| K+Q>=20 | 239 |

## Duplicates removed per class (top 15)

கௌ:6, ரோ:4, டௌ:3, ளி:3, ச்:3, ல்:3, யொ:3, உ:2, கீ:2, ஙே:2, ஞோ:2, ணொ:2, ஞு:2, ஙா:2, ஞூ:2


# Split summary

Images in manifest: 14202 across 247 classes; cached sizes: [48, 64, 96]

## Protocol A (image split, all classes)

| pool | images | classes | min/class | median/class |
|---|---|---|---|---|
| train | 9936 | 247 | 2 | 32 |
| val | 2133 | 247 | 1 | 7 |
| test | 2133 | 247 | 1 | 7 |

Source-group share per pool: train: other 78%, numeric 9%, screenshot 5%, img_date 3%, whatsapp 2%, resized 2%, scanned 1%, pxl 0%; val: other 78%, numeric 9%, screenshot 5%, img_date 3%, whatsapp 2%, resized 2%, scanned 1%, pxl 0%; test: other 77%, numeric 10%, screenshot 6%, img_date 2%, whatsapp 2%, resized 2%, scanned 1%, pxl 0%

## Protocol B (class split)

| part | classes | images | by category |
|---|---|---|---|
| base | 173 | 9825 | {'consonant-vowel': 152, 'consonant': 12, 'vowel': 8, 'aytham': 1} |
| val | 37 | 2271 | {'consonant-vowel': 32, 'consonant': 3, 'vowel': 2} |
| novel | 37 | 2106 | {'consonant-vowel': 32, 'consonant': 3, 'vowel': 2} |
| hard_novel | 31 | 1643 | {'consonant-vowel': 31} |

Small classes forced into base: ணு, ணூ, ணெ, ணே, ழ, ழா, ழி, ழீ
Novel classes: அ ஏ கே ஙை ச சி ஞ ஞா ஞி ஞௌ டி டெ ட் ணௌ தி தோ த் நௌ னீ னூ ப பௌ மி மௌ யீ ரே ற றோ லி லீ லெ லெள ளீ ழொ ழ் வா வௌ
Val classes: எ ஒ கீ கு கை கொ சா சூ சொ ஞோ டீ து தெள ந நு நெ நை னெ னை பி பொ ப் மை மோ ம் யி யே யோ ரொ ர் றூ றெ றொ லா ளி ளு ழே
