import React from 'react';
import { XIcon, BookIcon } from './Icons.jsx';

const SOURCES_DATA = [
  {
    id: 'Table 1',
    title: 'Antibiotics for non-pregnant women aged 16 years and over',
    page: '12',
    first: 'Nitrofurantoin (100mg modified-release bd for 3 days), Trimethoprim (200mg bd for 3 days)',
    second: 'Nitrofurantoin, Pivmecillinam (400mg then 200mg tds), Fosfomycin (3g single dose)',
  },
  {
    id: 'Table 2',
    title: 'Antibiotics for pregnant women aged 12 years and over',
    page: '13',
    first: 'Nitrofurantoin (100mg modified-release bd for 7 days — avoid at term)',
    second: 'Amoxicillin (500mg tds for 7 days if culture susceptible), Cefalexin (500mg bd for 7 days)',
  },
  {
    id: 'Table 3',
    title: 'Antibiotics for men aged 16 years and over',
    page: '14',
    first: 'Trimethoprim (200mg bd for 7 days), Nitrofurantoin (100mg modified-release bd for 7 days)',
    second: 'Consider alternative diagnoses and consult pyelonephritis/prostatitis guidelines',
  },
  {
    id: 'Table 4',
    title: 'Antibiotics for children and young people under 16 years',
    page: '15, 16',
    first: 'Trimethoprim (dose by age/weight for 3 days), Nitrofurantoin (dose by age/weight for 3 days)',
    second: 'Amoxicillin, Cefalexin based on microbiological results',
  },
  {
    id: 'Rec 1.1.2',
    title: 'Self-Care and Symptom Advice',
    page: '5, 10',
    first: 'Paracetamol or Ibuprofen for pain relief; advice on hydration',
    second: 'No evidence found for cranberry products or urine alkalinising agents',
  },
];

export default function SourcesModal({ isOpen, onClose }) {
  if (!isOpen) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-box" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div className="modal-header-title">
            <BookIcon className="w-4 h-4" />
            <span>NICE NG109 Guideline Sources</span>
          </div>
          <button type="button" className="btn-close" onClick={onClose}>
            <XIcon className="w-4 h-4" />
          </button>
        </div>

        <div className="modal-body">
          <p className="modal-intro">
            This CDS assistant is indexed against the official NICE guideline NG109: <em>Urinary tract infection (lower): antimicrobial prescribing</em>.
          </p>

          <div className="sources-list">
            {SOURCES_DATA.map((src) => (
              <div key={src.id} className="source-card">
                <div className="source-card-header">
                  <span className="source-badge">{src.id}</span>
                  <h4 className="source-title">{src.title}</h4>
                  <span className="source-page">Page {src.page}</span>
                </div>
                <div className="source-card-body">
                  <div className="choice-row">
                    <strong>First choice:</strong> <span>{src.first}</span>
                  </div>
                  <div className="choice-row">
                    <strong>Second choice:</strong> <span>{src.second}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="modal-footer">
          <button type="button" className="btn-done" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>
  );
}